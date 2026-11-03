"""
Read an outfit photo: find the person, crop each garment, and label it.

1. A pretrained pose model (YOLOv8-pose, COCO keypoints) finds the main person
   and their shoulders, hips, knees and ankles.
2. The keypoints split the body into regions: upper body, lower body, feet,
   plus a shoulder-to-knee crop used to tell dresses apart from top + bottom.
3. Each crop is embedded with the same CNN as the catalog and labelled by
   linear classifiers trained on the catalog embeddings
   (fashion/garment_classifier.py). Only categories that can appear in that
   region compete: upper → top/outerwear/dress, lower → bottom/dress,
   feet → shoes.
4. Color comes from the crop's pixels (dominant cluster near the middle of the
   garment), mapped onto the catalog palette.

The result says what the person is wearing, which outfit slots are empty, and
which catalog items look most like each piece.

    python -m fashion.outfit photo.jpg
    python -m fashion.outfit photo.jpg --save-crops out/ --json
"""

import argparse
import json
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from fashion.garment_classifier import GarmentClassifier
from fashion.index import CatalogIndex

POSE_WEIGHTS = "yolov8n-pose.pt"   # ultralytics fetches this from its GitHub releases

# COCO keypoint ids
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

KEYPOINT_CONF = 0.5

# Which catalog categories may compete for each body region.
REGION_CATEGORIES = {
    "upper": ["top", "outerwear", "dress"],
    "lower": ["bottom", "dress"],
    "feet":  ["shoes"],
    "full":  ["dress", "top", "bottom"],
}

# Slots a complete outfit should fill. A dress fills both top and bottom, and
# a jacket is assumed to cover a top.
OUTFIT_SLOTS = ["top", "bottom", "shoes", "outerwear", "accessories"]
REQUIRED_SLOTS = {"top", "bottom", "shoes"}

@dataclass
class Piece:
    region: str
    category: str
    article: str
    color: str
    confidence: float                     # classifier probability of the category
    box: tuple                            # (x0, y0, x1, y1) in the original photo
    category_scores: dict = field(default_factory=dict)
    similar: list = field(default_factory=list)   # [(item_id, score), ...]


@dataclass
class OutfitAnalysis:
    pieces: list
    missing: list          # outfit slots with nothing detected
    not_in_frame: list     # regions the photo cuts off (e.g. feet)
    person_box: tuple

    def to_dict(self):
        return asdict(self)


# ── Person + region detection ─────────────────────────────────────────────────

class PoseDetector:
    def __init__(self, weights=POSE_WEIGHTS):
        from ultralytics import YOLO
        self.model = YOLO(weights)

    def main_person(self, image):
        """Return (box xyxy, keypoints [17, 3] as x, y, conf) of the largest person, or None."""
        result = self.model(np.asarray(image.convert("RGB"))[:, :, ::-1], verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(areas.argmax())
        kpts = result.keypoints.data[i].cpu().numpy()
        return tuple(float(v) for v in boxes[i]), kpts


def _visible(kpts, *ids):
    return all(kpts[i, 2] >= KEYPOINT_CONF for i in ids)


def _span(kpts, ids):
    pts = np.array([kpts[i, :2] for i in ids if kpts[i, 2] >= KEYPOINT_CONF])
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def body_regions(person_box, kpts, image_size):
    """
    Turn pose keypoints into garment crop boxes.

    Returns (regions, not_in_frame) where regions maps region name → box.
    A region is only produced when the keypoints that bound it are visible;
    a waist-up photo still gets an upper-body crop.
    """
    W, H = image_size
    px0, py0, px1, py1 = person_box
    pw, ph = px1 - px0, py1 - py0
    regions, not_in_frame = {}, []

    def clamp(x0, y0, x1, y1):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        return (int(x0), int(y0), int(x1), int(y1)) if x1 - x0 > 8 and y1 - y0 > 8 else None

    shoulders = _visible(kpts, L_SHOULDER, R_SHOULDER)
    hips = _visible(kpts, L_HIP, R_HIP)
    knees = _visible(kpts, L_KNEE) or _visible(kpts, R_KNEE)
    ankles = _visible(kpts, L_ANKLE) or _visible(kpts, R_ANKLE)

    if shoulders and hips:
        x0, _, x1, _ = _span(kpts, [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_HIP, R_HIP])
        top_y = min(kpts[L_SHOULDER, 1], kpts[R_SHOULDER, 1])
        hip_y = max(kpts[L_HIP, 1], kpts[R_HIP, 1])
        pad = 0.08 * pw
        regions["upper"] = clamp(x0 - pad, top_y - 0.06 * ph, x1 + pad, hip_y + 0.04 * ph)
    elif shoulders:
        # Waist-up photo: the torso runs from the shoulders to the bottom of the frame.
        x0, _, x1, _ = _span(kpts, [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW])
        top_y = min(kpts[L_SHOULDER, 1], kpts[R_SHOULDER, 1])
        pad = 0.08 * pw
        regions["upper"] = clamp(x0 - pad, top_y - 0.04 * ph, x1 + pad, py1)

    if hips and (knees or ankles):
        leg_ids = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
        x0, _, x1, y1 = _span(kpts, leg_ids)
        hip_y = min(kpts[L_HIP, 1], kpts[R_HIP, 1])
        pad = 0.08 * pw
        regions["lower"] = clamp(x0 - pad, hip_y - 0.02 * ph, x1 + pad, y1 + 0.01 * ph)
    elif shoulders and not hips:
        not_in_frame.append("lower")

    if ankles:
        x0, y0, x1, y1 = _span(kpts, [L_ANKLE, R_ANKLE])
        pad_x, pad_down = 0.18 * pw, 0.07 * ph
        regions["feet"] = clamp(x0 - pad_x, y0 - 0.03 * ph, x1 + pad_x, max(y1 + pad_down, py1))
    else:
        not_in_frame.append("feet")

    if shoulders and knees:
        x0, _, x1, _ = _span(kpts, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE])
        top_y = min(kpts[L_SHOULDER, 1], kpts[R_SHOULDER, 1])
        knee_y = max(kpts[i, 1] for i in (L_KNEE, R_KNEE) if kpts[i, 2] >= KEYPOINT_CONF)
        pad = 0.1 * pw
        regions["full"] = clamp(x0 - pad, top_y - 0.06 * ph, x1 + pad, knee_y + 0.04 * ph)

    return {k: v for k, v in regions.items() if v}, not_in_frame


# ── Color ─────────────────────────────────────────────────────────────────────

def color_name(rgb):
    """
    Map an RGB value onto the catalog palette using hue, saturation and value.

    Working in HSV rather than nearest-RGB keeps a cream jacket in shadow
    "beige" instead of "gray", and dark denim "navy" instead of "black".
    """
    r, g, b = (float(c) / 255 for c in rgb)
    hi, lo = max(r, g, b), min(r, g, b)
    v, s = hi, (hi - lo) / hi if hi > 0 else 0.0
    if hi == lo:
        h = 0.0
    elif hi == r:
        h = (60 * (g - b) / (hi - lo)) % 360
    elif hi == g:
        h = 60 * (b - r) / (hi - lo) + 120
    else:
        h = 60 * (r - g) / (hi - lo) + 240

    if v < 0.16:
        return "navy" if 190 <= h < 260 and s > 0.45 else "black"
    if s < 0.12 or (s < 0.18 and v > 0.85):
        return "white" if v > 0.82 else "gray" if v > 0.3 else "black"
    if h < 15 or h >= 345:
        return "pink" if v > 0.75 and s < 0.45 else "red"
    if h < 45:
        if s < 0.45:
            return "beige" if v > 0.55 else "brown"
        return "brown" if v < 0.55 else "orange"
    if h < 70:
        if s < 0.4 and v > 0.6:
            return "beige"
        return "yellow" if v > 0.5 else "green"
    if h < 170:
        return "green"
    if h < 200:
        return "blue"
    if h < 250:
        return "navy" if v < 0.4 else "blue"
    if h < 290:
        return "purple"
    return "purple" if v < 0.45 else "pink"


def _kmeans(pixels, k, seed=0, iters=10):
    rng = np.random.default_rng(seed)
    centres = pixels[rng.choice(len(pixels), k, replace=False)]
    for _ in range(iters):
        labels = ((pixels[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        centres = np.array([pixels[labels == j].mean(0) if (labels == j).any() else centres[j]
                            for j in range(k)])
    return centres, np.bincount(labels, minlength=k)


def dominant_color(crop, skip_background=False, k=4, background_dist=40.0):
    """
    Name the main color of a garment crop.

    Pixels from the middle of the crop are clustered and the biggest cluster
    wins. With skip_background, clusters matching the crop's border are
    skipped first: leg and feet crops show a lot of pavement between and
    around the legs. Torso crops don't, and there the garment itself often
    reaches the border, so they don't skip anything.
    """
    small = np.asarray(crop.convert("RGB").resize((64, 64)), dtype=np.float32)
    border = np.concatenate([small[:4].reshape(-1, 3), small[-4:].reshape(-1, 3),
                             small[:, :4].reshape(-1, 3), small[:, -4:].reshape(-1, 3)])
    background, _ = _kmeans(border, 2)

    centre = small[13:51, 16:48].reshape(-1, 3)
    centres, sizes = _kmeans(centre, k)
    if not skip_background:
        return color_name(centres[sizes.argmax()])
    for j in np.argsort(-sizes):
        if np.sqrt(((background - centres[j]) ** 2).sum(1)).min() > background_dist:
            return color_name(centres[j])
    return color_name(centres[sizes.argmax()])


# ── Labelling crops ───────────────────────────────────────────────────────────

class OutfitReader:
    def __init__(self, index=None, detector=None, classifier=None):
        self.index = index or CatalogIndex()
        self.detector = detector or PoseDetector()
        self.classifier = classifier or GarmentClassifier()
        if self.classifier.backbone != self.index.manifest["backbone"]:
            raise ValueError(
                f"garment classifier was trained on {self.classifier.backbone} embeddings but the "
                f"index uses {self.index.manifest['backbone']}; rerun "
                f"python -m fashion.garment_classifier --backbone {self.index.manifest['backbone']}")

    def read(self, image):
        image = image.convert("RGB")
        found = self.detector.main_person(image)
        if found is None:
            return OutfitAnalysis([], list(OUTFIT_SLOTS), [], None)
        person_box, kpts = found
        regions, not_in_frame = body_regions(person_box, kpts, image.size)

        crops = {name: image.crop(box) for name, box in regions.items()}
        vectors = dict(zip(crops, self.index.embed(list(crops.values())))) if crops else {}
        probs = {name: self.classifier.category_probs(vec, REGION_CATEGORIES[name])
                 for name, vec in vectors.items()}

        def best(name):
            return max(probs[name], key=probs[name].get) if name in probs else None

        # A dress shows up as "dress" on the shoulder-to-knee crop and on at
        # least one of the upper/lower crops; then it replaces top + bottom.
        is_dress = best("full") == "dress" and "dress" in (best("upper"), best("lower"))

        pieces = []
        if is_dress:
            pieces.append(self._piece("full", "dress", regions, crops, vectors, probs))
        else:
            for name in ("upper", "lower"):
                if name not in probs:
                    continue
                # Without dress agreement from the full-length crop, pick the
                # region's best non-dress category.
                options = {c: p for c, p in probs[name].items() if c != "dress"}
                category = max(options, key=options.get)
                pieces.append(self._piece(name, category, regions, crops, vectors, probs))
        if "feet" in probs:
            pieces.append(self._piece("feet", "shoes", regions, crops, vectors, probs))

        worn = {p.category for p in pieces}
        if "dress" in worn:
            worn |= {"top", "bottom"}
        if "outerwear" in worn:
            # Something is under the jacket; the photo just can't show what.
            worn.add("top")
        missing = [slot for slot in OUTFIT_SLOTS if slot not in worn]
        return OutfitAnalysis(pieces, missing, not_in_frame, person_box)

    def _piece(self, region, category, regions, crops, vectors, probs, k=5):
        vec = vectors[region]
        return Piece(
            region=region,
            category=category,
            article=self.classifier.article(vec, category),
            color=dominant_color(crops[region], skip_background=region in ("lower", "feet")),
            confidence=round(probs[region].get(category, 0.0), 4),
            box=regions[region],
            category_scores={c: round(p, 4) for c, p in probs[region].items()},
            similar=[(item_id, round(s, 4)) for item_id, s in self.index.search(vec, category, k)[0]],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("photo")
    parser.add_argument("--save-crops", metavar="DIR", help="write each garment crop as a JPEG")
    parser.add_argument("--json", action="store_true", help="print the analysis as JSON")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="random CNN weights; only for testing the pipeline offline")
    args = parser.parse_args()

    reader = OutfitReader(CatalogIndex(pretrained=not args.no_pretrained))
    image = Image.open(args.photo)
    analysis = reader.read(image)

    if args.save_crops:
        from pathlib import Path
        out = Path(args.save_crops)
        out.mkdir(parents=True, exist_ok=True)
        for p in analysis.pieces:
            image.convert("RGB").crop(p.box).save(out / f"{p.region}_{p.category}.jpg")

    if args.json:
        print(json.dumps(analysis.to_dict(), indent=2))
        return

    if not analysis.pieces:
        print("No person found in the photo.")
        return
    print("Wearing:")
    for p in analysis.pieces:
        print(f"  {p.region:<6} {p.color} {p.article} ({p.category}, p={p.confidence:.2f})")
        for item_id, score in p.similar[:3]:
            print(f"           {score:.3f}  {reader.index.catalog.loc[item_id, 'name']}")
    print("Missing:", ", ".join(analysis.missing) or "nothing")
    if analysis.not_in_frame:
        print("Not in frame:", ", ".join(analysis.not_in_frame))


if __name__ == "__main__":
    main()
