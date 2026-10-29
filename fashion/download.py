"""
Download the Fashion Product Images (Small) dataset.

Uses the Hugging Face mirror (ashraq/fashion-product-images-small), which needs
no login, and unpacks it into the same layout as the Kaggle original:

    data/raw/styles.csv
    data/raw/images/<id>.jpg

If you already have the Kaggle zip (paramaggarwal/fashion-product-images-small),
just unzip it so that styles.csv and images/ sit under data/raw/ and skip this.

    python -m fashion.download
    python -m fashion.download --source rows   # if *.hf.co file storage is blocked

The default source pulls the parquet shards from the Hub's file storage
(*.hf.co). `--source rows` instead pages through the dataset viewer API on
datasets-server.huggingface.co, which is slower but only needs huggingface.co
hosts.
"""

import argparse
import io
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from fashion import RAW_DIR

HF_REPO = "ashraq/fashion-product-images-small"
ROWS_API = "https://datasets-server.huggingface.co/rows"
ROWS_PAGE = 100  # the API's maximum page size

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


def _fetch(url, retries=5):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _rows_page(repo_id, offset):
    query = urllib.parse.urlencode({
        "dataset": repo_id, "config": "default", "split": "train",
        "offset": offset, "length": ROWS_PAGE,
    })
    return json.loads(_fetch(f"{ROWS_API}?{query}"))


def download_rows(out_dir=RAW_DIR, repo_id=HF_REPO, workers=16):
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    def fetch_page(offset):
        # Image URLs are signed per request, so fetch them right after the page.
        page = _rows_page(repo_id, offset)
        records = []
        for item in page["rows"]:
            record = dict(item["row"])
            image = record.pop("image")
            img_path = images_dir / f"{record['id']}.jpg"
            if not img_path.exists():
                data = _fetch(image["src"])
                Image.open(io.BytesIO(data)).convert("RGB").save(img_path, quality=95)
            records.append({c: record.get(c) for c in META_COLUMNS})
        return records, page["num_rows_total"]

    first, total = fetch_page(0)
    rows = list(first)
    offsets = range(ROWS_PAGE, total, ROWS_PAGE)
    with ThreadPoolExecutor(workers) as pool:
        for records, _ in tqdm(pool.map(fetch_page, offsets), total=len(offsets), desc="pages"):
            rows.extend(records)

    styles = pd.DataFrame(rows, columns=META_COLUMNS)
    styles.to_csv(out_dir / "styles.csv", index=False)
    print(f"Wrote {len(styles)} products to {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=HF_REPO)
    parser.add_argument("--source", choices=["hub", "rows"], default="hub")
    args = parser.parse_args()
    if args.source == "rows":
        download_rows(repo_id=args.repo)
    else:
        download(repo_id=args.repo)


if __name__ == "__main__":
    main()
