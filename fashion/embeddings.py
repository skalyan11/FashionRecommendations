"""
Embed every catalog image with a pretrained image encoder (CPU-friendly).

Images are padded to a square with white (the catalog's background) instead of
center-cropped, so shoe tips and dress hems survive. CNN backbones drop their
classifier and pool the last conv stage. FashionCLIP uses its own 512-d projected image embedding. Every vector is
L2-normalized, so the dot product between two vectors is their cosine
similarity. Output:

    data/embeddings/<backbone>.npy        float32 [n_items, dim]
    data/embeddings/<backbone>_ids.txt    item_id per row
    data/embeddings/<backbone>_pca.npz    PCA mean + projection (only with --pca-dim)

    python -m fashion.embeddings --backbone convnext_tiny
"""

import argparse

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from fashion import EMBEDDINGS_DIR, RAW_DIR
from fashion.catalog import load_catalog

IMAGE_SIZE = 224
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)
FASHION_CLIP_ID = "patrickjohncyh/fashion-clip"

# name → (feature dim, pooling)
BACKBONES = {
    "fashion_clip":    (512,  "clip"),
    "resnet50":        (2048, "gem"),
    "convnext_tiny":   (768,  "avg"),
    "efficientnet_b0": (1280, "gem"),
}
# Best of the torchvision CNNs on python -m fashion.evaluate. FashionCLIP needs
# the Hub's file storage (*.hf.co), which some networks block.
DEFAULT_BACKBONE = "convnext_tiny"


class GeM(torch.nn.Module):
    """Generalized-mean pooling; p=1 is average pooling, large p tends to max."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p, self.eps = p, eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)


class FashionCLIPImage(torch.nn.Module):
    """FashionCLIP's image tower plus projection: pixels → 512-d embedding."""

    def __init__(self, pretrained=True):
        super().__init__()
        from transformers import CLIPConfig, CLIPModel
        if pretrained:
            self.clip = CLIPModel.from_pretrained(FASHION_CLIP_ID)
        else:
            self.clip = CLIPModel(CLIPConfig(projection_dim=512))

    def forward(self, x):
        out = self.clip.get_image_features(pixel_values=x)
        return out if torch.is_tensor(out) else out.pooler_output


def pad_to_square(img, fill=(255, 255, 255)):
    w, h = img.size
    side = max(w, h)
    return ImageOps.expand(img, ((side - w) // 2, (side - h) // 2,
                                 (side - w + 1) // 2, (side - h + 1) // 2), fill=fill)


def make_transform(mean, std):
    return transforms.Compose([
        transforms.Lambda(pad_to_square),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_backbone(name=DEFAULT_BACKBONE, pretrained=True):
    """Return (feature extractor in eval mode, preprocessing transform, raw dim)."""
    dim, pooling = BACKBONES[name]
    if name == "fashion_clip":
        return FashionCLIPImage(pretrained).eval(), make_transform(CLIP_MEAN, CLIP_STD), dim

    if name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        trunk = torch.nn.Sequential(*list(net.children())[:-2])     # up to layer4
    elif name == "convnext_tiny":
        net = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        trunk = net.features
    else:
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        trunk = net.features

    if pooling == "gem":
        model = torch.nn.Sequential(trunk, GeM())
    else:
        # ConvNeXt's last stage isn't post-ReLU, so keep its own avgpool + LayerNorm.
        model = torch.nn.Sequential(trunk, net.avgpool, net.classifier[0], net.classifier[1])
    model = model.to(memory_format=torch.channels_last)
    return model.eval(), make_transform(IMAGENET_MEAN, IMAGENET_STD), dim


# ── PCA whitening ─────────────────────────────────────────────────────────────

def fit_pca(vectors, dim):
    """Fit PCA whitening on L2-normalized vectors. Returns {"mean", "proj"}."""
    mean = vectors.mean(axis=0)
    _, s, vt = np.linalg.svd(vectors - mean, full_matrices=False)
    scale = s[:dim] / np.sqrt(len(vectors) - 1)
    proj = (vt[:dim].T / (scale + 1e-6)).astype(np.float32)
    return {"mean": mean.astype(np.float32), "proj": proj}


def apply_pca(vectors, pca):
    out = (vectors - pca["mean"]) @ pca["proj"]
    return out / np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-12)


def load_pca(backbone, out_dir=EMBEDDINGS_DIR):
    path = out_dir / f"{backbone}_pca.npz"
    return dict(np.load(path)) if path.exists() else None


# ── Embedding ─────────────────────────────────────────────────────────────────

class ImageFolderDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


@torch.inference_mode()
def embed_images(images, model, transform, batch_size=64, pca=None):
    """Embed a list of PIL images or file paths. Returns float32 [n, dim]."""
    if images and not isinstance(images[0], Image.Image):
        loader = DataLoader(
            ImageFolderDataset(images, transform),
            batch_size=batch_size,
            num_workers=min(4, torch.get_num_threads()),
        )
        batches = loader
    else:
        batches = [
            torch.stack([transform(img.convert("RGB")) for img in images[i:i + batch_size]])
            for i in range(0, len(images), batch_size)
        ]

    out = []
    for batch in tqdm(batches, desc="embedding", disable=len(images) <= batch_size):
        feats = model(batch.contiguous(memory_format=torch.channels_last))
        out.append(torch.nn.functional.normalize(feats, dim=1).numpy())
    vectors = np.concatenate(out).astype(np.float32)
    return apply_pca(vectors, pca).astype(np.float32) if pca is not None else vectors


def embed_catalog(backbone=DEFAULT_BACKBONE, pretrained=True, batch_size=64, pca_dim=None,
                  raw_dir=RAW_DIR, out_dir=EMBEDDINGS_DIR):
    catalog = load_catalog()
    model, transform, _ = load_backbone(backbone, pretrained)

    paths = [str(raw_dir / p) for p in catalog["image_path"]]
    vectors = embed_images(paths, model, transform, batch_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    if pca_dim:
        pca = fit_pca(vectors, pca_dim)
        np.savez(out_dir / f"{backbone}_pca.npz", **pca)
        vectors = apply_pca(vectors, pca).astype(np.float32)

    np.save(out_dir / f"{backbone}.npy", vectors)
    (out_dir / f"{backbone}_ids.txt").write_text("\n".join(catalog["item_id"]) + "\n")
    print(f"Embedded {len(vectors)} images → {out_dir / backbone}.npy  shape={vectors.shape}")
    return vectors


def fingerprint(vectors):
    """
    Short hash of a set of catalog embeddings. Models trained on one embedding
    run record it, so they can refuse vectors from a different run, even one
    with the same backbone name but different preprocessing.
    """
    import hashlib
    return hashlib.sha1(np.ascontiguousarray(vectors[:64], dtype=np.float32).tobytes()).hexdigest()[:12]


def load_embeddings(backbone=DEFAULT_BACKBONE, out_dir=EMBEDDINGS_DIR):
    vectors = np.load(out_dir / f"{backbone}.npy")
    ids = (out_dir / f"{backbone}_ids.txt").read_text().split()
    return ids, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", choices=BACKBONES, default=DEFAULT_BACKBONE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="random weights; only for testing the pipeline offline")
    parser.add_argument("--pca-dim", type=int,
                        help="PCA-whiten to this many dims (off by default; it lowered precision in evaluation)")
    args = parser.parse_args()
    embed_catalog(args.backbone, not args.no_pretrained, args.batch_size, args.pca_dim)


if __name__ == "__main__":
    main()
