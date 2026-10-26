"""
Normalize the raw product metadata into catalog.csv.

The output uses the same vocabulary as items.csv (category, style, color) so the
existing rule-based scoring in outfit_recommender.py can run over real products.
Products that aren't wearable outfit pieces (innerwear, personal care, ...) or
whose image is missing are dropped.

    python -m fashion.catalog
"""

import argparse

import pandas as pd

from fashion import CATALOG_PATH, RAW_DIR

# ── Category mapping ──────────────────────────────────────────────────────────
# articleType wins over subCategory, so jackets land in outerwear rather than top.

ARTICLE_CATEGORY = {
    "Jackets":     "outerwear",
    "Blazers":     "outerwear",
    "Coats":       "outerwear",
    "Rain Jacket": "outerwear",
    "Shrug":       "outerwear",
    "Waistcoat":   "outerwear",
    "Nehru Jackets": "outerwear",
    "Sweatshirts": "top",
    "Sweaters":    "top",
    "Dresses":     "dress",
    "Jumpsuit":    "dress",
    "Sarees":      "dress",
    "Lehenga Choli": "dress",
    "Rompers":     "dress",
}

SUBCATEGORY_CATEGORY = {
    "Topwear":    "top",
    "Bottomwear": "bottom",
    "Dress":      "dress",
    "Saree":      "dress",
    "Shoes":      "shoes",
    "Sandal":     "shoes",
    "Flip Flops": "shoes",
    "Bags":       "accessories",
    "Watches":    "accessories",
    "Belts":      "accessories",
    "Eyewear":    "accessories",
    "Jewellery":  "accessories",
    "Headwear":   "accessories",
    "Scarves":    "accessories",
    "Ties":       "accessories",
    "Wallets":    "accessories",
    "Cufflinks":  "accessories",
    "Gloves":     "accessories",
    "Mufflers":   "accessories",
    "Stoles":     "accessories",
    "Accessories": "accessories",
}

# ── Style mapping ─────────────────────────────────────────────────────────────
# The dataset's "usage" field is the closest thing it has to a style.

USAGE_STYLE = {
    "Casual":       "casual",
    "Formal":       "formal",
    "Smart Casual": "smart casual",
    "Sports":       "sporty",
    "Party":        "party",
    "Ethnic":       "ethnic",
    "Travel":       "practical",
    "Home":         "casual",
}

# ── Color mapping ─────────────────────────────────────────────────────────────
# Collapse the dataset's ~45 colour names onto the palette items.csv uses.

COLOR_MAP = {
    "Navy Blue": "navy", "Blue": "blue", "Turquoise Blue": "blue", "Teal": "blue",
    "Grey": "gray", "Grey Melange": "gray", "Steel": "gray", "Charcoal": "gray",
    "Silver": "gray", "Metallic": "gray",
    "Off White": "white", "Cream": "beige", "Beige": "beige", "Khaki": "beige",
    "Skin": "beige", "Nude": "beige", "Tan": "brown", "Coffee Brown": "brown",
    "Bronze": "brown", "Copper": "brown", "Mushroom Brown": "brown",
    "Taupe": "brown", "Rust": "brown",
    "Olive": "green", "Sea Green": "green", "Lime Green": "green",
    "Fluorescent Green": "neon",
    "Maroon": "red", "Burgundy": "red", "Red": "red", "Rose": "pink",
    "Magenta": "pink", "Peach": "pink", "Pink": "pink",
    "Lavender": "purple", "Mauve": "purple", "Purple": "purple",
    "Mustard": "yellow", "Gold": "yellow", "Yellow": "yellow",
    "Orange": "orange", "Multi": "mixed",
}


def _category(row):
    return ARTICLE_CATEGORY.get(row["articleType"]) or SUBCATEGORY_CATEGORY.get(row["subCategory"])


def _color(raw):
    if not isinstance(raw, str):
        return "unknown"
    return COLOR_MAP.get(raw, raw.lower())


def build_catalog(raw_dir=RAW_DIR, out_path=CATALOG_PATH):
    # A handful of rows in the Kaggle styles.csv have unquoted commas in the
    # product name; skip them rather than misparse them.
    styles = pd.read_csv(raw_dir / "styles.csv", on_bad_lines="skip")
    images_dir = raw_dir / "images"

    catalog = pd.DataFrame({
        "item_id":    "p" + styles["id"].astype(str),
        "name":       styles["productDisplayName"].fillna("").str.strip(),
        "category":   styles.apply(_category, axis=1),
        "style":      styles["usage"].map(USAGE_STYLE).fillna("casual"),
        "color":      styles["baseColour"].map(_color),
        "gender":     styles["gender"].str.lower(),
        "season":     styles["season"].str.lower(),
        "article":    styles["articleType"],
        "image_path": ("images/" + styles["id"].astype(str) + ".jpg"),
    })

    before = len(catalog)
    catalog = catalog[catalog["category"].notna()]
    has_image = catalog["image_path"].map(lambda p: (raw_dir / p).exists())
    catalog = catalog[has_image].reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_path, index=False)

    print(f"Kept {len(catalog)} of {before} products → {out_path}")
    print(catalog["category"].value_counts().to_string())
    return catalog


def load_catalog(path=CATALOG_PATH):
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args()
    build_catalog()


if __name__ == "__main__":
    main()
