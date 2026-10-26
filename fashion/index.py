"""
Build FAISS similarity indexes over the catalog embeddings and search them.

One exact inner-product index is built per category (top, bottom, shoes, ...)
plus one over everything, so later steps can ask "which shoes look most like
this?" without filtering after the fact.

    python -m fashion.index build
    python -m fashion.index search path/to/photo.jpg --category shoes -k 5
"""

import argparse
import json

import faiss
import numpy as np

from fashion import INDEX_DIR
from fashion.catalog import load_catalog
from fashion.embeddings import load_backbone, load_embeddings, embed_images

ALL = "all"


def build_indexes(backbone="resnet50", out_dir=INDEX_DIR):
    ids, vectors = load_embeddings(backbone)
    catalog = load_catalog().set_index("item_id").loc[ids]

    out_dir.mkdir(parents=True, exist_ok=True)
    groups = {ALL: np.arange(len(ids))}
    for category, rows in catalog.reset_index().groupby("category").groups.items():
        groups[category] = np.asarray(rows)

    manifest = {"backbone": backbone, "dim": int(vectors.shape[1]), "categories": {}}
    for category, rows in groups.items():
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors[rows])
        faiss.write_index(index, str(out_dir / f"{category}.faiss"))
        (out_dir / f"{category}_ids.txt").write_text("\n".join(ids[i] for i in rows) + "\n")
        manifest["categories"][category] = len(rows)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


class CatalogIndex:
    """Loads the per-category indexes and answers nearest-neighbour queries."""

    def __init__(self, out_dir=INDEX_DIR, pretrained=True):
        self.manifest = json.loads((out_dir / "manifest.json").read_text())
        self.indexes, self.ids = {}, {}
        for category in self.manifest["categories"]:
            self.indexes[category] = faiss.read_index(str(out_dir / f"{category}.faiss"))
            self.ids[category] = (out_dir / f"{category}_ids.txt").read_text().split()
        self.catalog = load_catalog().set_index("item_id")
        self._model = None
        self._pretrained = pretrained

    def embed(self, images):
        if self._model is None:
            self._model = load_backbone(self.manifest["backbone"], self._pretrained)[:2]
        return embed_images(images, *self._model)

    def search(self, vectors, category=ALL, k=10, exclude=()):
        """vectors: float32 [n, dim]. Returns, per query, a list of (item_id, score)."""
        index, ids = self.indexes[category], self.ids[category]
        scores, rows = index.search(np.atleast_2d(vectors), k + len(exclude))
        results = []
        for score_row, idx_row in zip(scores, rows):
            hits = [(ids[i], float(s)) for s, i in zip(score_row, idx_row)
                    if i >= 0 and ids[i] not in exclude]
            results.append(hits[:k])
        return results

    def similar_to_item(self, item_id, category=ALL, k=10, backbone=None):
        ids, vectors = load_embeddings(backbone or self.manifest["backbone"])
        vec = vectors[ids.index(item_id)]
        return self.search(vec, category, k, exclude={item_id})[0]


def _print_hits(index, hits):
    for item_id, score in hits:
        row = index.catalog.loc[item_id]
        print(f"  {score:.3f}  {item_id:<8} {row['category']:<11} {row['color']:<8} {row['name']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--backbone", default="resnet50")

    s = sub.add_parser("search")
    s.add_argument("query", help="an image path, or a catalog item_id like p15970")
    s.add_argument("--category", default=ALL)
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--no-pretrained", action="store_true")

    args = parser.parse_args()
    if args.cmd == "build":
        build_indexes(args.backbone)
        return

    index = CatalogIndex(pretrained=not args.no_pretrained)
    if args.query in index.catalog.index:
        hits = index.similar_to_item(args.query, args.category, args.k)
    else:
        from PIL import Image
        hits = index.search(index.embed([Image.open(args.query)]), args.category, args.k)[0]
    _print_hits(index, hits)


if __name__ == "__main__":
    main()
