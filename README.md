# FashionRecommendations


Workflow
python generate_users.py → users.csv
python generate_interactions.py → items.csv + interactions.csv



Builds fashion recommendations based on user input, existing data. Uses ML, python, and web scraping.


## Image catalog (step 1)

Real product photos are embedded with a pretrained CNN and indexed for
similarity search. Everything runs on CPU.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m fashion.download              # ~44k product images → data/raw/
python -m fashion.catalog               # data/raw/styles.csv → data/catalog.csv
python -m fashion.embeddings            # ResNet50 features → data/embeddings/ (~45 min on 4 CPU cores)
python -m fashion.index build           # FAISS index per category → data/index/

python -m fashion.index search p15970 --category shoes -k 5   # similar to a catalog item
python -m fashion.index search my_photo.jpg -k 5              # similar to any image
```

The dataset is Fashion Product Images (Small), pulled from its Hugging Face
mirror. If you have the Kaggle zip instead, unzip it into `data/raw/` and skip
`fashion.download`. `catalog.csv` uses the same `category`, `style` and `color`
vocabulary as `items.csv`, with `outerwear` and `dress` added as categories.
