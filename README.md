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
mirror. If the Hub's file storage (`*.hf.co`) is blocked on your network, run
`python -m fashion.download --source rows` instead; it pages through the
dataset viewer API on `datasets-server.huggingface.co` (about 10 minutes). If
you have the Kaggle zip instead, unzip it into `data/raw/` and skip
`fashion.download`. `catalog.csv` uses the same `category`, `style` and `color`
vocabulary as `items.csv`, with `outerwear` and `dress` added as categories.


## Reading an outfit photo (step 2)

`fashion.outfit` takes a photo of a person and works out what they're wearing:

1. **Find the person.** A pretrained YOLOv8 pose model finds the largest
   person and their shoulders, hips, knees and ankles. The weights
   (`yolov8n-pose.pt`, 6.5 MB) download from the Ultralytics GitHub releases
   on first run.
2. **Crop the garments.** The keypoints give an upper-body, lower-body, feet
   and shoulder-to-knee crop. A waist-up photo still gets an upper-body crop,
   and anything cut off is reported as "not in frame".
3. **Label each crop.** Crops are embedded with the step 1 ResNet50 and
   labelled by linear classifiers trained on the catalog embeddings. Only
   categories that make sense for the region compete, and a dress is only
   called when the full-length crop and the upper or lower crop agree.
4. **Name the color** from the crop's pixels in HSV. Leg and feet crops skip
   pavement-colored clusters.
5. **Report what's missing** (top, bottom, shoes, outerwear, accessories) and
   the closest catalog items to each piece. Step 3 builds recommendations
   from these.

```bash
python -m fashion.garment_classifier     # train the labelling heads (~1.5 min on CPU)
python -m fashion.outfit photo.jpg        # what's worn, what's missing, look-alikes
python -m fashion.outfit photo.jpg --json --save-crops crops/
```

The classifiers score 93% on outfit category and 78% on article type (52
types), on a held-out 20% of the catalog. Those are product photos on white
backgrounds, so expect lower accuracy on photos of people.

Ultralytics is AGPL-3.0 licensed. That's fine for a class or portfolio
project, but it matters if this is ever shipped as a closed-source service.
