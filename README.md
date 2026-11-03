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
python -m fashion.embeddings            # ConvNeXt-Tiny features → data/embeddings/ (~30 min on 4 CPU cores)
python -m fashion.index build           # FAISS index per category → data/index/

python -m fashion.index search p15970 --category shoes -k 5   # similar to a catalog item
python -m fashion.index search my_photo.jpg -k 5              # similar to any image
```

`--backbone` picks the image encoder for both `fashion.embeddings` and
`fashion.index build`: `convnext_tiny` (default), `resnet50`,
`efficientnet_b0` (about 2.5x faster, nearly as accurate) or `fashion_clip`.
FashionCLIP's weights come from the Hub's file storage (`*.hf.co`), so it only
works where that host is allowed. Images are padded to a white square before
resizing to 224 px, so whole garments stay in frame.

To compare encoders on a random 5k-item sample (precision@5 of neighbours that
share the query's category, article type and style):

```bash
python -m fashion.evaluate convnext_tiny resnet50 efficientnet_b0 --n 5000
```

The dataset is Fashion Product Images (Small), pulled from its Hugging Face
mirror. If the Hub's file storage (`*.hf.co`) is blocked on your network, run
`python -m fashion.download --source rows` instead; it pages through the
dataset viewer API on `datasets-server.huggingface.co` (about 10 minutes). If
you have the Kaggle zip instead, unzip it into `data/raw/` and skip
`fashion.download`. `catalog.csv` uses the same `category`, `style` and `color`
vocabulary as `items.csv`, with `outerwear` and `dress` added as categories.
