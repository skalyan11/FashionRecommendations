"""
Train the outfit-photo labelling heads on photos of people (DeepFashion).

The catalog classifier (fashion/garment_classifier.py) learned from product
shots, but step 2 asks it about crops of a person's body. On real photos that
mismatch cost a lot: 52% on bottom-vs-dress for the lower body and under 50%
on article type. These heads learn from the same kind of crop they will see.

Data: DeepFashion In-shop full-body and front photos, via the Hugging Face
mirror SaffalPoosh/deepFashion-with-masks. Each photo sells one product, and
its cloth_type (Dresses, Pants, Jackets_Coats, ...) labels the body region
that product covers. Photos whose caption contradicts that label (a "Dresses"
photo captioned "a blue shirt and plaid skirt") are dropped. Photos are split
85/15 by product id, so no garment appears in both training and validation.

    python -m fashion.people download     # ~14k photos, ~300 MB, ~10 min
    python -m fashion.people crops        # pose crops, ~15 min on CPU
    python -m fashion.people embed        # ~30 min on 4 CPUs
    python -m fashion.people train        # seconds; prints validation accuracy

The heads are saved to data/models/photo_classifier.npz as plain arrays.
"""

import argparse
import hashlib
import io
import json
import random
import re
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm

from fashion import DATA_DIR
from fashion.download import _fetch
from fashion.embeddings import DEFAULT_BACKBONE

HF_REPO = "SaffalPoosh/deepFashion-with-masks"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PEOPLE_DIR = DATA_DIR / "people"
MODEL_PATH = DATA_DIR / "models" / "photo_classifier.npz"
VAL_PERCENT = 15

# Front views of the most common products are capped so t-shirts don't
# swamp the rest; full-body views are all kept.
FRONT_CAP = {"Tees_Tanks": 1500, "Blouses_Shirts": 1000, "Dresses": 1000}

# cloth_type -> (region the product covers, outfit category, article type)
CLOTH = {
    "Tees_Tanks": ("upper", "top", "Tshirts"),
    "Graphic_Tees": ("upper", "top", "Tshirts"),
    "Blouses_Shirts": ("upper", "top", "Shirts"),
    "Shirts_Polos": ("upper", "top", "Shirts"),
    "Sweaters": ("upper", "top", "Sweaters"),
    "Sweatshirts_Hoodies": ("upper", "top", "Sweatshirts"),
    "Cardigans": ("upper", "outerwear", "Cardigans"),
    "Jackets_Coats": ("upper", "outerwear", "Jackets"),
    "Jackets_Vests": ("upper", "outerwear", "Jackets"),
    "Dresses": ("full", "dress", "Dresses"),
    "Rompers_Jumpsuits": ("full", "dress", "Jumpsuit"),
    "Pants": ("lower", "bottom", "Trousers"),
    "Denim": ("lower", "bottom", "Jeans"),
    "Shorts": ("lower", "bottom", "Shorts"),
    "Skirts": ("lower", "bottom", "Skirts"),
    "Leggings": ("lower", "bottom", "Leggings"),
}
DRESSY = ("dress", "romper", "jumpsuit", "overall", "playsuit")
OUTER = ("jacket", "coat", "cardigan", "blazer", "vest", "kimono")
BOTTOM = ("pants", "jeans", "shorts", "skirt", "leggings", "trousers")

# Which crop shows the labelled product, for the article heads.
ARTICLE_REGION = {"top": "upper", "outerwear": "upper", "bottom": "lower", "dress": "full"}
EMBED_REGIONS = ("upper", "lower", "full")


def _mentions(caption, *words):
    return any(re.search(rf"\b{w}", caption) for w in words)


def label(cloth_type, caption):
    """Region, category and article for one photo, or None when unusable."""
    if cloth_type not in CLOTH:
        return None
    region, category, article = CLOTH[cloth_type]
    c = caption.lower()
    if category == "dress" and not _mentions(c, *DRESSY) and _mentions(c, *BOTTOM):
        return None
    # A top worn under a jacket isn't what the upper crop shows.
    if category == "top" and (_mentions(c, *OUTER) or (_mentions(c, "dress") and not _mentions(c, *BOTTOM))):
        return None
    if category == "bottom" and _mentions(c, "dress") and not _mentions(c, *BOTTOM):
        return None
    if cloth_type == "Denim":
        if _mentions(c, "jacket", "shirt dress", "overall"):
            return None
        if _mentions(c, "shorts"):
            article = "Shorts"
        elif _mentions(c, "skirt"):
            article = "Skirts"
    out = {"region": region, "category": category, "article": article}
    # The caption also says what's worn below a top; usable for the
    # bottom-vs-dress head (category only, training only).
    if category in ("top", "outerwear") and _mentions(c, *BOTTOM) and not _mentions(c, *DRESSY):
        out["lower_category"] = "bottom"
    return out


def split_of(pid):
    return "val" if int(hashlib.md5(pid.encode()).hexdigest(), 16) % 100 < VAL_PERCENT else "train"


# ── download ──────────────────────────────────────────────────────────────────

def download(out_dir=PEOPLE_DIR, workers=16, seed=0):
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    def page(offset):
        query = urllib.parse.urlencode({"dataset": HF_REPO, "config": "default", "split": "train",
                                        "offset": offset, "length": 100})
        return json.loads(_fetch(f"{ROWS_API}?{query}"))

    first = page(0)
    offsets = range(0, first["num_rows_total"], 100)
    rows = []
    with ThreadPoolExecutor(workers) as pool:
        for p in tqdm(pool.map(page, offsets), total=len(offsets), desc="metadata"):
            for item in p["rows"]:
                r = item["row"]
                rows.append({"idx": item["row_idx"], "pid": r["pid"], "pose": r["pose"],
                             "gender": r["gender"], "cloth": r["cloth_type"], "caption": r["caption"],
                             "src": r["images"]["src"]})

    random.Random(seed).shuffle(rows)
    seen, keep = Counter(), []
    for r in rows:
        view = r["pose"].split("_")[-1]
        if r["cloth"] not in CLOTH or view not in ("full", "front"):
            continue
        if view == "front" and r["cloth"] in FRONT_CAP:
            if seen[r["cloth"]] >= FRONT_CAP[r["cloth"]]:
                continue
            seen[r["cloth"]] += 1
        keep.append(r)

    def fetch(r):
        path = images_dir / f"{r['idx']}.jpg"
        if not path.exists():
            Image.open(io.BytesIO(_fetch(r["src"]))).convert("RGB").save(path, quality=95)

    with ThreadPoolExecutor(workers) as pool:
        list(tqdm(pool.map(fetch, keep), total=len(keep), desc="images"))
    for r in keep:
        r.pop("src")
    (out_dir / "meta.json").write_text(json.dumps(keep))
    print(f"Wrote {len(keep)} photos to {out_dir}")


# ── crops ─────────────────────────────────────────────────────────────────────

def crops(out_dir=PEOPLE_DIR):
    from fashion.outfit import PoseDetector, body_regions

    meta = json.loads((out_dir / "meta.json").read_text())
    detector, photos = PoseDetector(), []
    for r in tqdm(meta, desc="pose"):
        lab = label(r["cloth"], r["caption"])
        if lab is None:
            continue
        image = Image.open(out_dir / "images" / f"{r['idx']}.jpg").convert("RGB")
        found = detector.main_person(image)
        if found is None:
            continue
        regions, _ = body_regions(found[0], found[1], image.size)
        photos.append({"idx": r["idx"], "pid": r["pid"], "pose": r["pose"], "cloth": r["cloth"],
                       "caption": r["caption"], "split": split_of(r["pid"]), "regions": regions, **lab})
    (out_dir / "photos.json").write_text(json.dumps(photos, default=float))
    print(f"{len(photos)} labelled photos:", Counter((p["split"], p["category"]) for p in photos))


# ── embed ─────────────────────────────────────────────────────────────────────

def embed(backbone=DEFAULT_BACKBONE, out_dir=PEOPLE_DIR, batch_size=64):
    from fashion.embeddings import embed_images, load_backbone

    model, transform, _ = load_backbone(backbone)
    photos = json.loads((out_dir / "photos.json").read_text())
    keys, vectors, batch = [], [], []

    def flush():
        if batch:
            vectors.append(embed_images([b[1] for b in batch], model, transform, batch_size))
            keys.extend(b[0] for b in batch)
            batch.clear()

    for p in tqdm(photos, desc=f"embedding ({backbone})"):
        image = Image.open(out_dir / "images" / f"{p['idx']}.jpg").convert("RGB")
        for region in EMBED_REGIONS:
            if region in p["regions"]:
                batch.append((f"{p['idx']}:{region}", image.crop(tuple(p["regions"][region]))))
        if len(batch) >= batch_size:
            flush()
    flush()
    np.savez(out_dir / f"crops_{backbone}.npz", keys=np.array(keys), vectors=np.concatenate(vectors))


# ── train ─────────────────────────────────────────────────────────────────────

def examples(photos, vectors):
    """(head, split, vector, label) for every usable crop."""
    by_idx = {p["idx"]: p for p in photos}
    for key, vec in vectors.items():
        idx, region = key.split(":")
        p = by_idx[int(idx)]
        cat, split = p["category"], p["split"]
        if region == "upper" and cat in ("top", "outerwear", "dress"):
            yield "upper", split, vec, cat
        if region == "lower" and cat in ("bottom", "dress"):
            yield "lower", split, vec, cat
        if region == "lower" and p.get("lower_category") and split == "train":
            yield "lower", split, vec, p["lower_category"]
        if region == "full" and cat == "dress":
            yield "full", split, vec, "dress"
        if region == "full" and (cat == "bottom" or p.get("lower_category")):
            yield "full", split, vec, "separates"
        if region == ARTICLE_REGION[cat]:
            yield f"article_{cat}", split, vec, p["article"]


def train(backbone=DEFAULT_BACKBONE, C=1.0, out_dir=PEOPLE_DIR, out_path=MODEL_PATH, refit=True):
    """Fit every head, print validation accuracy, then (with refit) refit on
    train + validation for the saved weights. refit=False keeps the
    validation photos unseen, for scoring the whole pipeline on them."""
    from sklearn.linear_model import LogisticRegression

    photos = json.loads((out_dir / "photos.json").read_text())
    z = np.load(out_dir / f"crops_{backbone}.npz")
    data = {}
    for head, split, vec, y in examples(photos, dict(zip(z["keys"].tolist(), z["vectors"]))):
        xs, ys = data.setdefault(head, {"train": ([], []), "val": ([], [])})[split]
        xs.append(vec)
        ys.append(y)

    arrays, report = {"backbone": np.array(backbone)}, {}
    for head, d in sorted(data.items()):
        X, y = np.array(d["train"][0]), np.array(d["train"][1])
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X, y)
        Xv, yv = np.array(d["val"][0]), np.array(d["val"][1])
        pred = clf.predict(Xv)
        report[head] = {
            "accuracy": float((pred == yv).mean()),
            "balanced_accuracy": float(np.mean([(pred[yv == c] == c).mean() for c in np.unique(yv)])),
            "n_train": int(len(y)), "n_val": int(len(yv)),
            "recall": {c: float((pred[yv == c] == c).mean()) for c in np.unique(yv)},
        }
        if refit:
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(
            np.concatenate([X, Xv]), np.concatenate([y, yv]))
        arrays[f"{head}_W"] = clf.coef_.astype(np.float32)
        arrays[f"{head}_b"] = clf.intercept_.astype(np.float32)
        arrays[f"{head}_labels"] = np.array(clf.classes_, dtype=str)
        r = report[head]
        print(f"  {head:<18} {r['accuracy']:.3f} (balanced {r['balanced_accuracy']:.3f})"
              f"  train {r['n_train']}, val {r['n_val']}")
    arrays["report"] = np.array(json.dumps(report))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    print(f"Saved {out_path}")
    return report


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


class PhotoClassifier:
    """Region heads trained on crops of people. Same embeddings as the index."""

    def __init__(self, path=MODEL_PATH):
        m = np.load(path, allow_pickle=False)
        self.backbone = str(m["backbone"])
        self.heads = {}
        for key in m.files:
            if key.endswith("_W"):
                head = key[:-2]
                self.heads[head] = (m[key], m[f"{head}_b"], m[f"{head}_labels"].tolist())
        self.report = json.loads(str(m["report"]))

    def _probs(self, head, vector):
        W, b, labels = self.heads[head]
        logits = vector @ W.T + b
        if len(labels) == 2 and W.shape[0] == 1:      # sklearn's binary layout
            logits = np.array([-logits[0], logits[0]]) / 2
        return dict(zip(labels, _softmax(logits).tolist()))

    def region_probs(self, region, vector):
        """Category probabilities for an upper/lower crop; dress/separates for full."""
        return self._probs(region, vector)

    def article(self, vector, category, exclude=()):
        head = f"article_{category}"
        if head not in self.heads:
            return None
        probs = self._probs(head, vector)
        allowed = {a: p for a, p in probs.items() if a not in exclude} or probs
        return max(allowed, key=allowed.get)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["download", "crops", "embed", "train"])
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("-C", type=float, default=1.0)
    parser.add_argument("--out", default=str(MODEL_PATH))
    parser.add_argument("--no-refit", action="store_true",
                        help="train on the training split only (to score the pipeline on validation)")
    args = parser.parse_args()
    if args.step == "download":
        download()
    elif args.step == "crops":
        crops()
    elif args.step == "embed":
        embed(args.backbone)
    else:
        from pathlib import Path
        train(args.backbone, args.C, out_path=Path(args.out), refit=not args.no_refit)


if __name__ == "__main__":
    main()
