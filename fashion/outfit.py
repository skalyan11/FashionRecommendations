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
NOSE, L_EYE, R_EYE = 0, 1, 2

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

# Bottoms that end above the shins (or mid-calf), and ones that cover them. When the shin
# probes show bare legs (or boots), long trousers are ruled out, and the
# other way round.
SHORT_BOTTOMS = {"Shorts", "Skirts", "Capris"}
LONG_BOTTOMS = {"Jeans", "Trousers", "Track Pants", "Leggings", "Jeggings", "Patiala",
                "Salwar", "Churidar", "Tights", "Stockings"}

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
    elif hips and min(H, py1) - max(kpts[L_HIP, 1], kpts[R_HIP, 1]) > 0.08 * ph:
        # Knees out of frame (a three-quarter shot): whatever shows below the
        # hips is still the bottom half of the outfit.
        x0, _, x1, _ = _span(kpts, [L_HIP, R_HIP])
        hip_y = min(kpts[L_HIP, 1], kpts[R_HIP, 1])
        pad = 0.2 * pw
        regions["lower"] = clamp(x0 - pad, hip_y - 0.02 * ph, x1 + pad, min(H, py1))
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


def garment_probes(kpts):
    """
    Small patches used to tell where one garment ends and the next begins.

    chest: the middle of the torso. thigh: just below the hips. shins: halfway
    down each lower leg. gap: between the knees, which is background for
    trousers and fabric for a long dress or skirt. A probe is left out when
    the keypoints that place it aren't visible.
    """
    probes = {}
    if not (_visible(kpts, L_SHOULDER, R_SHOULDER) and _visible(kpts, L_HIP, R_HIP)):
        return probes
    sx, sy = kpts[[L_SHOULDER, R_SHOULDER], :2].mean(0)
    hx, hy = kpts[[L_HIP, R_HIP], :2].mean(0)
    hw = max(abs(kpts[L_HIP, 0] - kpts[R_HIP, 0]), 0.25 * abs(hy - sy), 8.0)
    probes["chest"] = (sx - 0.5 * hw, sy + 0.3 * (hy - sy), sx + 0.5 * hw, sy + 0.7 * (hy - sy))

    knees = [k for k in (L_KNEE, R_KNEE) if kpts[k, 2] >= KEYPOINT_CONF]
    if not knees:
        return probes
    ky = float(np.mean([kpts[k, 1] for k in knees]))
    probes["thigh"] = (hx - 0.6 * hw, hy + 0.1 * (ky - hy), hx + 0.6 * hw, hy + 0.5 * (ky - hy))

    for side, (knee, ankle) in {"left": (L_KNEE, L_ANKLE), "right": (R_KNEE, R_ANKLE)}.items():
        if _visible(kpts, knee, ankle):
            (x0, y0), (x1, y1) = kpts[knee, :2], kpts[ankle, :2]
            cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, 0.2 * hw
            probes[f"shin_{side}"] = (cx - half, cy - 0.15 * (y1 - y0), cx + half, cy + 0.15 * (y1 - y0))
    if _visible(kpts, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE):
        kx = kpts[[L_KNEE, R_KNEE], 0].mean()
        half = max(0.25 * abs(kpts[L_KNEE, 0] - kpts[R_KNEE, 0]), 0.1 * hw)
        ay = kpts[[L_ANKLE, R_ANKLE], 1].mean()
        probes["gap"] = (kx - half, ky + 0.05 * (ay - ky), kx + half, ky + 0.3 * (ay - ky))
    return {k: tuple(int(round(v)) for v in box) for k, box in probes.items()
            if box[2] - box[0] >= 4 and box[3] - box[1] >= 4}


def _color_hist(crop, bins=8):
    a = np.asarray(crop.convert("RGB").resize((32, 32))).reshape(-1, 3) // (256 // bins)
    h = np.bincount((a[:, 0] * bins + a[:, 1]) * bins + a[:, 2], minlength=bins ** 3).astype(float)
    return h / h.sum()


def patch_similarity(crop_a, crop_b, vec_a, vec_b):
    """(CNN cosine, color-histogram intersection) between two patches."""
    cos = float(vec_a @ vec_b / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8))
    return cos, float(np.minimum(_color_hist(crop_a), _color_hist(crop_b)).sum())


# Two patches show the same garment when both texture (CNN) and colors agree.
# On 15 studio photos, dresses scored cos 0.67-0.80 / hist 0.63-0.76 between
# chest and thigh; separate tops and bottoms scored at most 0.56 / 0.39, apart
# from one pale-yellow top over white trousers.
SAME_GARMENT_COS = 0.6
SAME_GARMENT_HIST = 0.5
# Thigh-to-shin color overlap: trousers 0.44-0.92, bare legs or boots 0-0.19.
LEGS_COVERED_HIST = 0.35


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
    if v < 0.3 and s < 0.45 and not 190 <= h < 260:
        return "black"          # dark leather, shadowed black fabric
    # How much saturation still reads as "no color" depends on the hue: a
    # faintly warm off-white is beige/khaki, but a faintly pink gray is gray.
    neutral = 0.08 if 20 <= h < 70 else 0.2 if (h < 20 or h >= 330) else 0.12
    if s < neutral or (s < 0.18 and v > 0.85 and not 20 <= h < 70):
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


def _is_skin(rgb, skin_ref=None, ref_dist=30.0):
    """
    Does a cluster centre look like skin? Either it's close to the skin color
    sampled from this person's face (`skin_ref`), or it passes the classic
    RGB skin rule (Kovac et al.), which only covers lighter skin tones in good
    light; the face sample covers the rest.
    """
    if skin_ref is not None and np.sqrt(((np.asarray(rgb) - skin_ref) ** 2).sum()) < ref_dist:
        return True
    r, g, b = (float(c) for c in rgb)
    sat = (max(r, g, b) - min(r, g, b)) / max(r, g, b, 1.0)
    # The saturation band keeps pinkish grays and saturated tans/browns
    # (shorts, boots) from counting as skin.
    return (r > 95 and g > 40 and b > 20 and max(r, g, b) - min(r, g, b) > 15
            and abs(r - g) > 15 and r > g and r > b and 0.18 <= sat <= 0.58)


def face_skin_color(image, kpts):
    """Median color of a small patch under the eyes, or None if the face isn't visible."""
    if not _visible(kpts, NOSE, L_EYE, R_EYE):
        return None
    eye_dist = abs(kpts[L_EYE, 0] - kpts[R_EYE, 0])
    if eye_dist < 4:
        return None
    nx, ny = kpts[NOSE, :2]
    # Cheeks either side of the nose, below the eyes: skin, not hair or lips.
    boxes = [(nx - 0.9 * eye_dist, ny - 0.3 * eye_dist, nx - 0.3 * eye_dist, ny + 0.1 * eye_dist),
             (nx + 0.3 * eye_dist, ny - 0.3 * eye_dist, nx + 0.9 * eye_dist, ny + 0.1 * eye_dist)]
    pixels = [np.asarray(image.crop(tuple(int(v) for v in b)).convert("RGB"), np.float32).reshape(-1, 3)
              for b in boxes]
    pixels = [p for p in pixels if len(p)]
    if not pixels:
        return None
    ref = np.median(np.concatenate(pixels), 0)
    # On a small face the patch can land on hair or backdrop; only keep a
    # sample that is plausibly skin (warm hue, some but not much saturation).
    r, g, b = ref
    sat = (ref.max() - ref.min()) / max(ref.max(), 1.0)
    return ref if r > g > b and 0.12 <= sat <= 0.55 else None


def background_colors(image, k=3):
    """Color clusters along the photo's outer edge: the backdrop or street."""
    small = np.asarray(image.convert("RGB").resize((96, 96)), dtype=np.float32)
    edge = np.concatenate([small[:3].reshape(-1, 3), small[-3:].reshape(-1, 3),
                           small[:, :3].reshape(-1, 3), small[:, -3:].reshape(-1, 3)])
    return _kmeans(edge, k)[0]


def dominant_color(crop, background=None, sides=False, skin_ref=None, border=False, k=4,
                   background_dist=40.0, min_share=0.12):
    """
    Name the main color of a garment crop.

    Pixels from the middle of the crop are clustered and the biggest cluster
    wins, skipping clusters that look like skin or like the photo background
    (`background`, from background_colors) as long as a garment-looking
    cluster covers at least `min_share` of the pixels. Bare legs and arms
    otherwise outvote a short skirt or sandals.

    With sides=True the left and right thirds are sampled instead of the
    middle: an open jacket shows the shirt underneath in the middle and the
    jacket itself on the sides and sleeves. With border=True, colors along
    the crop's own edge count as background too: shoes stand on a floor
    that the photo's edge may not show.
    """
    small = np.asarray(crop.convert("RGB").resize((64, 64)), dtype=np.float32)
    if border:
        edge = np.concatenate([small[:4].reshape(-1, 3), small[-4:].reshape(-1, 3),
                               small[:, :4].reshape(-1, 3), small[:, -4:].reshape(-1, 3)])
        edge_bg = _kmeans(edge, 2)[0]
        background = edge_bg if background is None else np.concatenate([background, edge_bg])
    if sides:
        pixels = np.concatenate([small[8:56, 2:20].reshape(-1, 3), small[8:56, 44:62].reshape(-1, 3)])
    else:
        pixels = small[13:51, 16:48].reshape(-1, 3)
    centres, sizes = _kmeans(pixels, k)
    order = np.argsort(-sizes)

    def is_background(c):
        return background is not None and np.sqrt(((background - c) ** 2).sum(1)).min() < background_dist

    total = sizes.sum()
    for j in order:
        if (sizes[j] >= min_share * total and not _is_skin(centres[j], skin_ref)
                and not is_background(centres[j])):
            return color_name(centres[j])
    for j in order:          # nothing big enough: any garment-like cluster, then non-background
        if not _is_skin(centres[j], skin_ref) and not is_background(centres[j]):
            return color_name(centres[j])
    for j in order:
        if not is_background(centres[j]):
            return color_name(centres[j])
    return color_name(centres[order[0]])


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
        probes = garment_probes(kpts)
        background = background_colors(image)
        skin = face_skin_color(image, kpts)

        crops = {name: image.crop(box) for name, box in regions.items()}
        probe_crops = {name: image.crop(box) for name, box in probes.items()}
        batch = list(crops.values()) + list(probe_crops.values())
        embedded = self.index.embed(batch) if batch else []
        vectors = dict(zip(crops, embedded[:len(crops)]))
        probe_vecs = dict(zip(probe_crops, embedded[len(crops):]))
        probs = {name: self.classifier.category_probs(vec, REGION_CATEGORIES[name])
                 for name, vec in vectors.items()}

        def best(name):
            return max(probs[name], key=probs[name].get) if name in probs else None

        def same(a, b):
            if a not in probe_crops or b not in probe_crops:
                return None
            cos, hist = patch_similarity(probe_crops[a], probe_crops[b], probe_vecs[a], probe_vecs[b])
            return cos >= SAME_GARMENT_COS and hist >= SAME_GARMENT_HIST

        # Does the thigh garment carry on down the shins (trousers, maxi)?
        # Colors alone decide this: the thigh probe takes in the crotch, so
        # texture matches less well than on the torso.
        shins = [float(np.minimum(_color_hist(probe_crops["thigh"]), _color_hist(probe_crops[s])).sum())
                 for s in ("shin_left", "shin_right") if s in probe_crops and "thigh" in probe_crops]
        legs_covered = max(shins) >= LEGS_COVERED_HIST if shins else None
        gap_is_background = None
        if "gap" in probe_crops:
            gap_color = _kmeans(np.asarray(probe_crops["gap"].resize((16, 16)), np.float32).reshape(-1, 3), 2)
            centres, sizes = gap_color
            dist = np.sqrt(((background[None] - centres[:, None]) ** 2).sum(-1)).min(1)
            gap_is_background = bool(sizes[dist < 40].sum() > 0.5 * sizes.sum())

        # A dress is one garment from chest to thigh that stops above the
        # shins, or carries on to the ankles without a gap between the legs.
        # The catalog classifier rarely says "dress" on a crop (catalog dress
        # shots look nothing like a torso crop), so it's only a second route.
        one_piece = same("chest", "thigh")
        if one_piece and legs_covered is not None:
            looks_like_dress = (not legs_covered) or gap_is_background is False
        else:
            looks_like_dress = False
        is_dress = looks_like_dress or (
            best("full") == "dress" and "dress" in (best("upper"), best("lower")))

        pieces = []
        if is_dress:
            pieces.append(self._piece("full", "dress", regions, crops, vectors, probs, background, skin))
        else:
            for name in ("upper", "lower"):
                if name not in probs:
                    continue
                # Without dress agreement from the full-length crop, pick the
                # region's best non-dress category.
                options = {c: p for c, p in probs[name].items() if c != "dress"}
                category = max(options, key=options.get)
                exclude = ()
                if category == "bottom" and legs_covered is not None:
                    exclude = SHORT_BOTTOMS if legs_covered else LONG_BOTTOMS
                pieces.append(self._piece(name, category, regions, crops, vectors, probs,
                                          background, skin, exclude))
        if "feet" in probs:
            pieces.append(self._piece("feet", "shoes", regions, crops, vectors, probs, background, skin))

        worn = {p.category for p in pieces}
        if "dress" in worn:
            worn |= {"top", "bottom"}
        if "outerwear" in worn:
            # Something is under the jacket; the photo just can't show what.
            worn.add("top")
        missing = [slot for slot in OUTFIT_SLOTS if slot not in worn]
        return OutfitAnalysis(pieces, missing, not_in_frame, person_box)

    def _piece(self, region, category, regions, crops, vectors, probs, background=None,
               skin=None, exclude=(), k=5):
        vec = vectors[region]
        return Piece(
            region=region,
            category=category,
            article=self.classifier.article(vec, category, exclude),
            color=dominant_color(crops[region], background, sides=category == "outerwear",
                                 skin_ref=skin),
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
