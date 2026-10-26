"""Image-based fashion catalog: download, normalize, embed, and search."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CATALOG_PATH = DATA_DIR / "catalog.csv"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
INDEX_DIR = DATA_DIR / "index"
