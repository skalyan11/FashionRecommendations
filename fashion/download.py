"""
Download the Fashion Product Images (Small) dataset.

Uses the Hugging Face mirror (ashraq/fashion-product-images-small), which needs
no login, and unpacks it into the same layout as the Kaggle original:

    data/raw/styles.csv
    data/raw/images/<id>.jpg

If you already have the Kaggle zip (paramaggarwal/fashion-product-images-small),
just unzip it so that styles.csv and images/ sit under data/raw/ and skip this.

    python -m fashion.download
"""

import argparse
import io
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from fashion import RAW_DIR

HF_REPO = "ashraq/fashion-product-images-small"

META_COLUMNS = [
    "id", "gender", "masterCategory", "subCategory", "articleType",
    "baseColour", "season", "year", "usage", "productDisplayName",
]


def download(out_dir=RAW_DIR, repo_id=HF_REPO):
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["*.parquet"],
    )
    parquet_files = sorted(Path(snapshot).rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in {repo_id}")

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in parquet_files:
        df = pd.read_parquet(path)
        for record in tqdm(df.to_dict("records"), desc=path.name):
            image = record.pop("image")
            # HF image columns are stored as {"bytes": ..., "path": ...}
            data = image["bytes"] if isinstance(image, dict) else image
            img_path = images_dir / f"{record['id']}.jpg"
            if not img_path.exists():
                Image.open(io.BytesIO(data)).convert("RGB").save(img_path, quality=95)
            rows.append({c: record.get(c) for c in META_COLUMNS})

    styles = pd.DataFrame(rows, columns=META_COLUMNS)
    styles.to_csv(out_dir / "styles.csv", index=False)
    print(f"Wrote {len(styles)} products to {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=HF_REPO)
    args = parser.parse_args()
    download(repo_id=args.repo)


if __name__ == "__main__":
    main()
