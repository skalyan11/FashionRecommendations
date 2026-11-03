"""
Train linear classifiers that label a garment from its CNN embedding.

Two heads are trained on the catalog embeddings from step 1: one for the
outfit category (top, outerwear, dress, bottom, shoes, accessories) and one
for the article type (Jeans, Sweaters, Heels, ...). Both use class-balanced
weights, so small categories like outerwear aren't swamped by t-shirts. 20%
of the catalog is held out and the held-out accuracy is printed.

The weights are saved as plain arrays in data/models/garment_classifier.npz,
so loading them needs no pickle and no scikit-learn.

    python -m fashion.garment_classifier
"""

import argparse
import json

import numpy as np

from fashion import DATA_DIR
from fashion.catalog import load_catalog
from fashion.embeddings import load_embeddings

MODEL_PATH = DATA_DIR / "models" / "garment_classifier.npz"
MIN_ARTICLE_COUNT = 40


def _fit(X, y, C):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit(X, y)
    return clf


def _split(n, seed=0, holdout=0.2):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = int(n * (1 - holdout))
    return order[:cut], order[cut:]


def _per_class_report(y_true, y_pred):
    out = {}
    for label in sorted(set(y_true)):
        mask = y_true == label
        out[label] = {"n": int(mask.sum()), "recall": float((y_pred[mask] == label).mean())}
    return out


def train(backbone="resnet50", C=1.0, out_path=MODEL_PATH):
    ids, X = load_embeddings(backbone)
    catalog = load_catalog().set_index("item_id").loc[ids]
    categories = catalog["category"].to_numpy()
    articles = catalog["article"].to_numpy()

    train_rows, test_rows = _split(len(ids))
    report = {}

    cat_clf = _fit(X[train_rows], categories[train_rows], C)
    pred = cat_clf.predict(X[test_rows])
    report["category"] = {
        "accuracy": float((pred == categories[test_rows]).mean()),
        "per_class": _per_class_report(categories[test_rows], pred),
    }

    counts = catalog["article"].value_counts()
    keep = set(counts[counts >= MIN_ARTICLE_COUNT].index)
    art_train = train_rows[np.isin(articles[train_rows], list(keep))]
    art_test = test_rows[np.isin(articles[test_rows], list(keep))]
    art_clf = _fit(X[art_train], articles[art_train], C)
    pred = art_clf.predict(X[art_test])
    report["article"] = {
        "accuracy": float((pred == articles[art_test]).mean()),
        "n_classes": len(keep),
    }

    # Refit on everything before saving, now that the held-out numbers are known.
    cat_clf = _fit(X, categories, C)
    art_mask = np.isin(articles, list(keep))
    art_clf = _fit(X[art_mask], articles[art_mask], C)

    # Which category each article belongs to, so article predictions can be
    # restricted to the category that won.
    article_category = catalog.groupby("article")["category"].agg(lambda s: s.mode()[0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        backbone=backbone,
        cat_W=cat_clf.coef_, cat_b=cat_clf.intercept_, cat_labels=cat_clf.classes_.astype(str),
        art_W=art_clf.coef_, art_b=art_clf.intercept_, art_labels=art_clf.classes_.astype(str),
        art_category=np.array([article_category[a] for a in art_clf.classes_], dtype=str),
        report=json.dumps(report),
    )
    print(json.dumps(report, indent=2))
    print(f"Saved → {out_path}")
    return report


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class GarmentClassifier:
    def __init__(self, path=MODEL_PATH):
        m = np.load(path, allow_pickle=False)
        self.backbone = str(m["backbone"])
        self.cat_W, self.cat_b, self.cat_labels = m["cat_W"], m["cat_b"], m["cat_labels"].tolist()
        self.art_W, self.art_b, self.art_labels = m["art_W"], m["art_b"], m["art_labels"].tolist()
        self.art_category = m["art_category"].tolist()
        self.report = json.loads(str(m["report"]))

    def category_probs(self, vector, allowed=None):
        """Probability per category, renormalized over `allowed` if given."""
        logits = vector @ self.cat_W.T + self.cat_b
        labels = self.cat_labels
        if allowed is not None:
            keep = [i for i, c in enumerate(labels) if c in allowed]
            logits, labels = logits[keep], [labels[i] for i in keep]
        return dict(zip(labels, _softmax(logits).tolist()))

    def article(self, vector, category):
        """Most likely article type within the given category."""
        logits = vector @ self.art_W.T + self.art_b
        keep = [i for i, c in enumerate(self.art_category) if c == category]
        if not keep:
            return None
        return self.art_labels[keep[int(np.argmax(logits[keep]))]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("-C", type=float, default=1.0, help="inverse regularization strength")
    args = parser.parse_args()
    train(args.backbone, args.C)


if __name__ == "__main__":
    main()
