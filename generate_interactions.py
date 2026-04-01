"""
Generate user-item interactions by scoring every (user, item) pair.

Scoring logic
─────────────
  style_score    (+style_weight)   item.style in user.preferred_styles
  color_score    (+color_weight)   item.color in user.preferred_colors
                 (-color_weight/2) item.color in user.disliked_colors
  price_score    (+price_weight)   item.price within user.budget_range
  category_score (+category_weight) item.category in user.preferred_categories

Total score ∈ [0, 1].  Interaction thresholds:
  score >= PURCHASE_THRESHOLD  →  purchase  (implies a like too)
  score >= LIKE_THRESHOLD      →  like
  otherwise                    →  ignored

Reads: users.csv, sample_items.json
Writes: items.csv (catalog), interactions.csv
"""

import csv
import json
import random
from pathlib import Path

SEED = 42
LIKE_THRESHOLD = 0.45
PURCHASE_THRESHOLD = 0.70
LIST_SEP = "|"

random.seed(SEED)


def load_users_csv(path: Path) -> list[dict]:
    users = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            users.append(
                {
                    "user_id": row["user_id"],
                    "archetype": row["archetype"],
                    "preferred_styles": [s for s in row["preferred_styles"].split(LIST_SEP) if s],
                    "preferred_colors": [s for s in row["preferred_colors"].split(LIST_SEP) if s],
                    "disliked_colors": [s for s in row["disliked_colors"].split(LIST_SEP) if s],
                    "preferred_categories": [s for s in row["preferred_categories"].split(LIST_SEP) if s],
                    "budget_range": [int(row["budget_low"]), int(row["budget_high"])],
                    "style_weight": float(row["style_weight"]),
                    "color_weight": float(row["color_weight"]),
                    "price_weight": float(row["price_weight"]),
                    "category_weight": float(row["category_weight"]),
                }
            )
    return users


def write_items_csv(items: list, path: Path) -> None:
    fieldnames = ["item_id", "name", "category", "style", "color", "price"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for item in items:
            w.writerow(
                {
                    "item_id": item["id"],
                    "name": item["name"],
                    "category": item["category"],
                    "style": item["style"],
                    "color": item["color"],
                    "price": item["price"],
                }
            )


def write_interactions_csv(interactions: list, path: Path) -> None:
    fieldnames = ["user_id", "item_id", "item_name", "score", "event"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in interactions:
            w.writerow(row)


def score_item(user: dict, item: dict) -> float:
    score = 0.0

    # Style match
    if item["style"] in user["preferred_styles"]:
        score += user["style_weight"]

    # Color match / penalty
    if item["color"] in user["preferred_colors"]:
        score += user["color_weight"]
    elif item["color"] in user["disliked_colors"]:
        score -= user["color_weight"] / 2

    # Price within budget
    low, high = user["budget_range"]
    if low <= item["price"] <= high:
        score += user["price_weight"]
    elif item["price"] < low:
        # cheaper than budget floor — small partial credit
        score += user["price_weight"] * 0.5

    # Category match
    if item["category"] in user["preferred_categories"]:
        score += user["category_weight"]

    return round(max(0.0, min(1.0, score)), 4)


def generate_interactions(users: list, items: list) -> tuple[list, list]:
    interactions = []
    summaries = []

    for user in users:
        liked = []
        purchased = []
        ignored = []

        for item in items:
            base_score = score_item(user, item)
            # Add a small random nudge to simulate real-world noise
            noisy_score = base_score + random.gauss(0, 0.03)
            noisy_score = max(0.0, min(1.0, noisy_score))

            if noisy_score >= PURCHASE_THRESHOLD:
                event = "purchase"
                purchased.append(item["id"])
                liked.append(item["id"])
            elif noisy_score >= LIKE_THRESHOLD:
                event = "like"
                liked.append(item["id"])
            else:
                event = "ignored"
                ignored.append(item["id"])

            interactions.append(
                {
                    "user_id": user["user_id"],
                    "item_id": item["id"],
                    "item_name": item["name"],
                    "score": round(noisy_score, 4),
                    "event": event,
                }
            )

        summaries.append(
            {
                "user_id": user["user_id"],
                "archetype": user["archetype"],
                "liked_count": len(liked),
                "purchased_count": len(purchased),
                "ignored_count": len(ignored),
                "liked_items": liked,
                "purchased_items": purchased,
            }
        )

    return interactions, summaries


def print_example(users: list, items: list, interactions: list):
    """Print a human-readable example for the first user of each archetype."""
    seen_archetypes = set()

    print("\n── Example interactions ──")
    for user in users:
        if user["archetype"] in seen_archetypes:
            continue
        seen_archetypes.add(user["archetype"])

        user_interactions = [i for i in interactions if i["user_id"] == user["user_id"]]
        liked = [i for i in user_interactions if i["event"] in ("like", "purchase")]
        ignored = [i for i in user_interactions if i["event"] == "ignored"]

        print(f"\nUser: {user['user_id']}  ({user['archetype']})")
        print(f"  Budget: ${user['budget_range'][0]}–${user['budget_range'][1]}")
        print(f"  Liked ({len(liked)}):")
        for i in sorted(liked, key=lambda x: -x["score"])[:5]:
            print(f"    ✓  {i['item_name']}  [score={i['score']}]")
        print(f"  Ignored ({len(ignored)}):")
        for i in sorted(ignored, key=lambda x: x["score"])[:3]:
            print(f"    ✗  {i['item_name']}  [score={i['score']}]")


def main():
    users_path = Path("users.csv")
    items_path = Path("sample_items.json")
    items_csv_path = Path("items.csv")
    interactions_path = Path("interactions.csv")

    if not users_path.exists():
        print("users.csv not found — run generate_users.py first")
        return

    users = load_users_csv(users_path)
    with items_path.open(encoding="utf-8") as f:
        items = json.load(f)

    write_items_csv(items, items_csv_path)

    interactions, _summaries = generate_interactions(users, items)
    write_interactions_csv(interactions, interactions_path)

    total = len(interactions)
    likes = sum(1 for i in interactions if i["event"] == "like")
    purchases = sum(1 for i in interactions if i["event"] == "purchase")
    ignored = sum(1 for i in interactions if i["event"] == "ignored")

    print(f"Generated {total} interaction records ({len(users)} users × {len(items)} items)")
    print(f"  Purchases : {purchases:4d}  ({purchases/total*100:.1f}%)")
    print(f"  Likes     : {likes:4d}  ({likes/total*100:.1f}%)")
    print(f"  Ignored   : {ignored:4d}  ({ignored/total*100:.1f}%)")
    print(f"Saved → {items_csv_path}, {interactions_path}")

    print_example(users, items, interactions)


if __name__ == "__main__":
    main()
