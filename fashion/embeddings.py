"""
Embed every catalog image with a pretrained CNN (CPU-friendly).

The backbone's classifier head is removed and the pooled features are
L2-normalized, so the dot product between two vectors is their cosine
similarity. Output:

    data/embeddings/<backbone>.npy        float32 [n_items, dim]
    data/embeddings/<backbone>_ids.txt    item_id per row

    python -m fashion.embeddings --backbone resnet50
"""

import argparse

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

from fashion import EMBEDDINGS_DIR, RAW_DIR
from fashion.catalog import load_catalog

# name → (constructor, default weights, feature dim)
BACKBONES = {
    "resnet50":        (models.resnet50,        models.ResNet50_Weights.IMAGENET1K_V2,        2048),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
}


def load_backbone(name="resnet50", pretrained=True):
    """Return (feature extractor in eval mode, preprocessing transform, dim)."""
    build, weights, dim = BACKBONES[name]
    model = build(weights=weights if pretrained else None)

    # Drop the ImageNet classifier; keep the globally pooled features.
    if name.startswith("resnet"):
        model.fc = torch.nn.Identity()
    else:
        model.classifier = torch.nn.Identity()

    model.eval()
    return model, weights.transforms(), dim


class ImageFolderDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


@torch.inference_mode()
def embed_images(images, model, transform, batch_size=64):
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
        feats = model(batch)
        out.append(torch.nn.functional.normalize(feats, dim=1).numpy())
    return np.concatenate(out).astype(np.float32)


def embed_catalog(backbone="resnet50", pretrained=True, batch_size=64,
                  raw_dir=RAW_DIR, out_dir=EMBEDDINGS_DIR):
    catalog = load_catalog()
    model, transform, _ = load_backbone(backbone, pretrained)

    paths = [str(raw_dir / p) for p in catalog["image_path"]]
    vectors = embed_images(paths, model, transform, batch_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{backbone}.npy", vectors)
    (out_dir / f"{backbone}_ids.txt").write_text("\n".join(catalog["item_id"]) + "\n")
    print(f"Embedded {len(vectors)} images → {out_dir / backbone}.npy  shape={vectors.shape}")
    return vectors


def load_embeddings(backbone="resnet50", out_dir=EMBEDDINGS_DIR):
    vectors = np.load(out_dir / f"{backbone}.npy")
    ids = (out_dir / f"{backbone}_ids.txt").read_text().split()
    return ids, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", choices=BACKBONES, default="resnet50")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="random weights; only for testing the pipeline offline")
    args = parser.parse_args()
    embed_catalog(args.backbone, not args.no_pretrained, args.batch_size)


if __name__ == "__main__":
    main()
