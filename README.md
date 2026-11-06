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


## Reading an outfit photo (step 2)

`fashion.outfit` takes a photo of a person and works out what they're wearing:

1. **Find the person.** A pretrained YOLOv8 pose model finds the largest
   person and their shoulders, hips, knees and ankles. The weights
   (`yolov8n-pose.pt`, 6.5 MB) download from the Ultralytics GitHub releases
   on first run.
2. **Crop the garments.** The keypoints give an upper-body, lower-body, feet
   and shoulder-to-knee crop. A waist-up photo still gets an upper-body crop,
   a three-quarter shot (knees out of frame) still gets the bottom half, and
   anything cut off is reported as "not in frame".
3. **Label each crop.** Crops are embedded with the same encoder as the
   catalog index (ConvNeXt-Tiny by default) and labelled by linear heads
   trained on crops of people (`fashion/people.py`, below): upper body is
   top / outerwear / dress, lower body is bottom / dress, the
   shoulder-to-knee crop is dress / separates, and each category has an
   article head (Tshirts, Shirts, Sweaters, Sweatshirts, Jackets, Cardigans,
   Dresses, Jumpsuit, Trousers, Jeans, Shorts, Skirts, Leggings). Shoes, and
   everything when `data/models/photo_classifier.npz` is missing, fall back
   to the catalog-trained heads (`fashion/garment_classifier.py`). Both
   record the encoder they were trained on, and the reader refuses to run
   against an index built with a different one.
4. **Dress or separates.** The dress/separates head decides, and a shape test
   nudges borderline cases: probe patches on the chest, thigh, shins and
   between the knees check for one fabric from chest to thigh that stops
   above the shins (or reaches the ankles with no gap between the legs). A
   jacket or cardigan over a dress is reported as its own piece. Whether the
   shins are covered also rules out shorts/skirts or trousers when picking
   the bottom's article type.
5. **Name the color** from the crop's pixels in HSV, skipping clusters that
   look like skin (sampled from the person's face when visible) or like the
   photo's background. Jackets are colored from their sides and sleeves, not
   the shirt showing in the middle.
6. **Report what's missing** (top, bottom, shoes, outerwear, accessories) and
   the closest catalog items to each piece. Step 3 builds recommendations
   from these.

```bash
python -m fashion.garment_classifier     # catalog heads (shoes, fallback), ~1.5 min on CPU
python -m fashion.people download        # ~14k DeepFashion photos of people, ~300 MB
python -m fashion.people crops           # pose crops, ~15 min on CPU
python -m fashion.people embed           # ~30 min on 4 CPUs
python -m fashion.people train           # seconds; prints validation accuracy
python -m fashion.outfit photo.jpg        # what's worn, what's missing, look-alikes
python -m fashion.outfit photo.jpg --json --save-crops crops/
```

### Training on photos of people

The catalog heads learned from product shots, but here they label crops of
someone's body, and that mismatch was the main source of errors. The people
heads train on DeepFashion In-shop photos (Hugging Face mirror
`SaffalPoosh/deepFashion-with-masks`): each photo sells one product, and its
type (Dresses, Pants, Jackets_Coats, ...) labels the body region that product
covers. Photos whose caption contradicts the label are dropped, and photos
are split 85/15 by product so no garment is in both training and validation.

Whole pipeline on the 2,045 validation photos (heads trained on the training
split only):

| | before (catalog heads) | after (people heads) |
|---|---|---|
| right category for the labelled garment | 79.9% | 90.2% |
| right category and article | 43.7% | 71.7% |
| dresses found | 47.0% | 90.0% |
| non-dresses called a dress | 1.9% | 3.7% |

Per head, on validation crops (balanced accuracy in brackets): upper body
86.7% (88.1%), lower body 76.1%, dress vs separates 92.7%, articles: tops
73.9%, outerwear 87.7%, bottoms 83.7% (65.3%; few jeans and leggings),
dresses 90.6%.

CLIP was compared on the same crops. OpenAI's CLIP weights are on Hugging
Face file storage, which this environment can't reach, so this used
open_clip's ResNet-50 trained on CC12M (from GitHub). Zero-shot it did no
better than the catalog heads (upper 73%, lower 51%). With the same people
heads on its embeddings it was slightly ahead of ConvNeXt-Tiny (upper 88.2%,
dress vs separates 96.1%, articles within a point or two), not enough to
justify a second encoder next to the index's ConvNeXt-Tiny.

Ultralytics is AGPL-3.0 licensed. That's fine for a class or portfolio
project, but it matters if this is ever shipped as a closed-source service.
