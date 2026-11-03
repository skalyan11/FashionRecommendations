"""
Compare backbones on item-to-item retrieval over a random catalog subset.

For each item, its top-k neighbours (itself excluded) are checked against its
labels: same category, same article type (Shirts, Jeans, ...), and same style
(casual, formal, sporty, ...). Style precision is the best proxy here for
how well a backbone captures what makes pieces go together.

    python -m fashion.evaluate resnet50 convnext_tiny fashion_clip --n 5000
"""

import argparse
import time

import numpy as np

from fashion import RAW_DIR
from fashion.catalog import load_catalog
from fashion.embeddings import BACKBONES, apply_pca, embed_images, fit_pca, load_backbone

LABELS = ("category", "article", "style")


def precision_at_k(vectors, labels, k=5):
    """vectors: L2-normalized [n, dim]. Returns {label: precision@k}."""
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    return {name: float((values[top] == values[:, None]).mean())
            for name, values in labels.items()}


def evaluate(backbone, catalog, k=5, batch_size=64, pca_dim=None):
    model, transform, _ = load_backbone(backbone)
    paths = [str(RAW_DIR / p) for p in catalog["image_path"]]
    start = time.perf_counter()
    vectors = embed_images(paths, model, transform, batch_size)
    seconds = time.perf_counter() - start

    labels = {name: catalog[name].to_numpy() for name in LABELS}
    results = {"raw": precision_at_k(vectors, labels, k)}
    if pca_dim:
        results[f"pca{pca_dim}"] = precision_at_k(apply_pca(vectors, fit_pca(vectors, pca_dim)), labels, k)
    return results, len(paths) / seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("backbones", nargs="+", choices=BACKBONES)
    parser.add_argument("--n", type=int, default=5000, help="catalog items to sample")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pca-dim", type=int, help="also score PCA-whitened vectors")
    args = parser.parse_args()

    catalog = load_catalog()
    catalog = catalog.sample(min(args.n, len(catalog)), random_state=args.seed)

    print(f"{'backbone':<16} {'variant':<8} {'img/s':>6}  " +
          "  ".join(f"{name + '@' + str(args.k):>11}" for name in LABELS))
    for backbone in args.backbones:
        results, speed = evaluate(backbone, catalog, args.k, pca_dim=args.pca_dim)
        for variant, scores in results.items():
            print(f"{backbone:<16} {variant:<8} {speed:>6.1f}  " +
                  "  ".join(f"{scores[name]:>11.3f}" for name in LABELS))


if __name__ == "__main__":
    main()
