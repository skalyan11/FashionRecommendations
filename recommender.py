import csv
from collections import defaultdict


# ── Data loading ──────────────────────────────────────────────────────────────

def load_items(path="items.csv"):
    items = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            items[row["item_id"]] = {
                "item_id":  row["item_id"],
                "name":     row["name"],
                "category": row["category"],
                "style":    row["style"],
                "color":    row["color"],
                "price":    float(row["price"]),
            }
    return items


def load_interactions(path="interactions.csv"):
    """Return dict: user_id → list of {item_id, event, score}."""
    interactions = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            interactions[row["user_id"]].append({
                "item_id": row["item_id"],
                "event":   row["event"],
                "score":   float(row["score"]),
            })
    return interactions


# ── Step 1: Build user profile from interactions ───────────────────────────────

def build_user_profile(user_interactions, items):
    """
    Derive a preference profile from the items a user liked or purchased.
    Only 'like' and 'purchase' events count as positive signal.
    """
    style_counts    = defaultdict(int)
    color_counts    = defaultdict(int)
    category_counts = defaultdict(int)
    prices          = []

    engaged = [i for i in user_interactions if i["event"] in ("like", "purchase")]

    if not engaged:
        return None

    for interaction in engaged:
        item = items.get(interaction["item_id"])
        if item is None:
            continue
        style_counts[item["style"]] += 1
        color_counts[item["color"]] += 1
        category_counts[item["category"]] += 1
        prices.append(item["price"])

    def normalize(counts):
        total = sum(counts.values())
        return {k: round(v / total, 4) for k, v in counts.items()} if total else {}

    return {
        "style_pref":    normalize(style_counts),
        "color_pref":    normalize(color_counts),
        "category_pref": normalize(category_counts),
        "avg_price":     round(sum(prices) / len(prices), 2),
        "min_price":     min(prices),
        "max_price":     max(prices),
        # keep raw counts for explanations
        "_style_counts":    dict(style_counts),
        "_color_counts":    dict(color_counts),
        "_category_counts": dict(category_counts),
        "_engaged_count":   len(engaged),
    }


# ── Step 2: Score a single item for a user ────────────────────────────────────

WEIGHTS = {
    "style":    0.35,
    "color":    0.25,
    "category": 0.20,
    "price":    0.20,
}


def style_match(profile, item):
    return profile["style_pref"].get(item["style"], 0.0)


def color_match(profile, item):
    return profile["color_pref"].get(item["color"], 0.0)


def category_match(profile, item):
    return profile["category_pref"].get(item["category"], 0.0)


def price_match(profile, item):
    """
    Full credit if the item price is within the user's observed price range.
    Partial credit for items slightly outside that range (within 1.5x).
    """
    price     = item["price"]
    low, high = profile["min_price"], profile["max_price"]

    if low <= price <= high:
        return 1.0

    # Cheap but below observed range: partial credit
    if price < low:
        ratio = price / low  # 0 → 1 as price → low
        return max(0.0, ratio)

    # Expensive: decay as price exceeds upper end
    if price > high:
        ratio = high / price  # 1 → 0 as price increases
        return max(0.0, ratio)

    return 0.0


def score_item(profile, item):
    return (
        WEIGHTS["style"]    * style_match(profile, item)
        + WEIGHTS["color"]    * color_match(profile, item)
        + WEIGHTS["category"] * category_match(profile, item)
        + WEIGHTS["price"]    * price_match(profile, item)
    )


# ── Step 3: Recommend top-N items ─────────────────────────────────────────────

def _score_pool(pool, profile):
    return sorted(
        [(score_item(profile, item), item) for item in pool],
        key=lambda x: x[0],
        reverse=True,
    )


def recommend(user_id, interactions_by_user, items, top_n=10):
    """
    Candidate selection follows real-system semantics:

      Tier 1 — truly unseen items (no interaction at all)
      Tier 2 — liked but not purchased (user expressed interest, didn't buy)

    Purchased items are always excluded (already owned).
    Ignored items are excluded unless we exhaust Tier 1 + Tier 2.

    In production data users won't have seen every item, so Tier 1 alone fills
    the list.  This dataset is synthetic and covers all 47 items per user, so
    the tiered fallback keeps recommendations meaningful.
    """
    user_interactions = interactions_by_user.get(user_id, [])
    profile = build_user_profile(user_interactions, items)

    if profile is None:
        return []

    by_event = {}
    for i in user_interactions:
        by_event[i["item_id"]] = i["event"]

    purchased_ids = {iid for iid, ev in by_event.items() if ev == "purchase"}
    liked_ids     = {iid for iid, ev in by_event.items() if ev == "like"}
    ignored_ids   = {iid for iid, ev in by_event.items() if ev == "ignored"}
    seen_ids      = set(by_event)

    unseen   = [it for iid, it in items.items() if iid not in seen_ids]
    liked    = [it for iid, it in items.items() if iid in liked_ids]
    ignored  = [it for iid, it in items.items() if iid in ignored_ids]

    results = []

    # Tier 1: unseen items
    results.extend(_score_pool(unseen, profile))

    # Tier 2: fill with liked items if needed
    if len(results) < top_n:
        results.extend(_score_pool(liked, profile))

    # Tier 3 (last resort): ignored items — only if still short
    if len(results) < top_n:
        results.extend(_score_pool(ignored, profile))

    # Final sort across tiers, then truncate
    results.sort(key=lambda x: x[0], reverse=True)
    return [(score, item, profile) for score, item in results[:top_n]]


# ── Step 5: Explanations ──────────────────────────────────────────────────────

def explain(score, item, profile):
    reasons = []

    if style_match(profile, item) > 0:
        pct = round(profile["style_pref"][item["style"]] * 100)
        reasons.append(
            f"Matches your preferred style: {item['style']} "
            f"({pct}% of your liked/purchased items)"
        )

    if color_match(profile, item) > 0:
        pct = round(profile["color_pref"][item["color"]] * 100)
        reasons.append(
            f"Matches your color preference: {item['color']} "
            f"({pct}% of your liked/purchased items)"
        )

    if category_match(profile, item) > 0:
        pct = round(profile["category_pref"][item["category"]] * 100)
        reasons.append(
            f"Matches your category preference: {item['category']} "
            f"({pct}% of your liked/purchased items)"
        )

    pm = price_match(profile, item)
    if pm >= 1.0:
        reasons.append(
            f"Price ${item['price']:.0f} fits within your typical range "
            f"(${profile['min_price']:.0f}–${profile['max_price']:.0f})"
        )
    elif pm > 0:
        reasons.append(
            f"Price ${item['price']:.0f} is slightly outside your typical range "
            f"(${profile['min_price']:.0f}–${profile['max_price']:.0f})"
        )

    return reasons


# ── CLI demo ──────────────────────────────────────────────────────────────────

def print_recommendations(user_id, recommendations):
    if not recommendations:
        print(f"No recommendations for {user_id} (no engagement history).")
        return

    print(f"\n{'='*60}")
    print(f"  Top recommendations for {user_id}")
    print(f"{'='*60}")

    for rank, (score, item, profile) in enumerate(recommendations, 1):
        print(f"\n{rank}. {item['name']}  [score: {score:.3f}]")
        print(f"   Category: {item['category']} | Style: {item['style']} | "
              f"Color: {item['color']} | Price: ${item['price']:.0f}")
        for reason in explain(score, item, profile):
            print(f"   ✓ {reason}")


def print_profile_summary(user_id, interactions_by_user, items):
    profile = build_user_profile(interactions_by_user.get(user_id, []), items)
    if not profile:
        print(f"No profile for {user_id}")
        return

    print(f"\n{'─'*60}")
    print(f"  User profile: {user_id}")
    print(f"{'─'*60}")
    print(f"  Engaged with {profile['_engaged_count']} liked/purchased items")

    top_styles = sorted(profile["style_pref"].items(), key=lambda x: -x[1])[:3]
    print(f"  Style prefs:    " + ", ".join(f"{s} {v:.0%}" for s, v in top_styles))

    top_colors = sorted(profile["color_pref"].items(), key=lambda x: -x[1])[:3]
    print(f"  Color prefs:    " + ", ".join(f"{c} {v:.0%}" for c, v in top_colors))

    top_cats = sorted(profile["category_pref"].items(), key=lambda x: -x[1])[:4]
    print(f"  Category prefs: " + ", ".join(f"{c} {v:.0%}" for c, v in top_cats))

    print(f"  Avg price:      ${profile['avg_price']}")
    print(f"  Price range:    ${profile['min_price']:.0f}–${profile['max_price']:.0f}")


if __name__ == "__main__":
    items               = load_items()
    interactions_by_user = load_interactions()

    # Demo: show a sample of different archetype users
    demo_users = ["user_001", "user_008", "user_015", "user_022", "user_029"]

    for uid in demo_users:
        print_profile_summary(uid, interactions_by_user, items)
        recs = recommend(uid, interactions_by_user, items, top_n=5)
        print_recommendations(uid, recs)
