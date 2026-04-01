from itertools import product as cartesian_product

from recommender import (
    load_items,
    load_interactions,
    build_user_profile,
    score_item,
    _score_pool,
)


# ── Outfit templates ──────────────────────────────────────────────────────────
# Each template names the required categories (in display order).

OUTFIT_TEMPLATES = {
    "casual":         ["top", "bottom", "shoes"],
    "casual_layered": ["top", "bottom", "shoes", "accessories"],
    "smart_casual":   ["top", "bottom", "shoes", "accessories"],
    "sporty":         ["top", "bottom", "shoes"],
    "formal":         ["top", "bottom", "shoes", "accessories"],
}


# ── Contexts ──────────────────────────────────────────────────────────────────
# Each context maps to a template and the styles that "fit" that occasion.

CONTEXTS = {
    "casual_day": {
        "label":    "Casual Day Out",
        "styles":   {"casual", "minimalist", "streetwear", "preppy", "trendy", "expressive"},
        "template": "casual",
    },
    "cold_weather": {
        "label":    "Cold Weather",
        "styles":   {"practical", "casual", "sporty", "minimalist"},
        "template": "casual_layered",
    },
    "smart_casual": {
        "label":    "Smart Casual / Going Out",
        "styles":   {"smart casual", "preppy", "formal", "minimalist", "business"},
        "template": "smart_casual",
    },
    "formal": {
        "label":    "Formal Occasion",
        "styles":   {"formal", "business"},
        "template": "formal",
    },
    "workout": {
        "label":    "Workout / Gym",
        "styles":   {"sporty", "athleisure", "casual"},
        "template": "sporty",
    },
}


# ── Style compatibility matrix ────────────────────────────────────────────────
# Maps each style to the set of styles it pairs well with.

STYLE_COMPAT = {
    "minimalist":   {"minimalist", "casual", "formal", "smart casual", "business"},
    "casual":       {"casual", "minimalist", "sporty", "preppy", "smart casual", "practical"},
    "streetwear":   {"streetwear", "casual", "athleisure", "expressive", "trendy"},
    "preppy":       {"preppy", "smart casual", "casual", "formal"},
    "smart casual": {"smart casual", "preppy", "casual", "formal", "minimalist"},
    "formal":       {"formal", "business", "smart casual", "minimalist"},
    "business":     {"business", "formal", "smart casual", "minimalist"},
    "sporty":       {"sporty", "casual", "athleisure"},
    "athleisure":   {"athleisure", "sporty", "casual"},
    "trendy":       {"trendy", "expressive", "casual", "streetwear"},
    "expressive":   {"expressive", "trendy", "streetwear"},
    "practical":    {"practical", "casual", "sporty"},
}

NEUTRAL_COLORS = {"black", "white", "beige", "gray", "navy", "light blue", "mixed"}
BOLD_COLORS    = {"neon", "red", "yellow", "bright"}


# ── Compatibility sub-scores ──────────────────────────────────────────────────

def style_compatibility(outfit_items):
    """Fraction of item pairs whose styles are mutually compatible."""
    styles = [item["style"] for item in outfit_items]
    n = len(styles)
    if n < 2:
        return 1.0
    compatible = total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if styles[j] in STYLE_COMPAT.get(styles[i], set()):
                compatible += 1
    return compatible / total_pairs


def color_compatibility(outfit_items):
    """
    Penalize outfits that stack multiple bold/clashing colors.
    One accent on a neutral base is fine; two bold pieces clash.
    """
    bold_count = sum(1 for item in outfit_items if item["color"] in BOLD_COLORS)
    if bold_count == 0:
        return 1.00   # all neutrals — clean palette
    if bold_count == 1:
        return 0.85   # single accent — works
    if bold_count == 2:
        return 0.40   # two bolds — risky clash
    return 0.10       # three or more — very clashing


def outfit_compatibility(outfit_items):
    """Combined style + color compatibility (equal weight)."""
    return 0.5 * style_compatibility(outfit_items) + 0.5 * color_compatibility(outfit_items)


# ── Context match ─────────────────────────────────────────────────────────────

def context_match(outfit_items, context_key):
    """Fraction of outfit items whose style fits the given context."""
    context_styles = CONTEXTS[context_key]["styles"]
    matches = sum(1 for item in outfit_items if item["style"] in context_styles)
    return matches / len(outfit_items)


# ── Budget fit ────────────────────────────────────────────────────────────────

def budget_fit(profile, outfit_items):
    """
    Compare total outfit price against the user's per-item ceiling scaled by
    outfit size.  Full credit within range; smooth decay if over.
    """
    total   = sum(item["price"] for item in outfit_items)
    ceiling = profile["max_price"] * len(outfit_items)
    if total <= ceiling:
        return 1.0
    return max(0.0, ceiling / total)


# ── User preference match ─────────────────────────────────────────────────────

def user_preference_match(profile, outfit_items):
    """Average per-item score across the outfit."""
    return sum(score_item(profile, item) for item in outfit_items) / len(outfit_items)


# ── Outfit scorer ─────────────────────────────────────────────────────────────

OUTFIT_WEIGHTS = {
    "user_pref":     0.45,
    "compatibility": 0.30,
    "context":       0.15,
    "budget":        0.10,
}


def score_outfit(profile, outfit_items, context_key):
    return (
        OUTFIT_WEIGHTS["user_pref"]      * user_preference_match(profile, outfit_items)
        + OUTFIT_WEIGHTS["compatibility"] * outfit_compatibility(outfit_items)
        + OUTFIT_WEIGHTS["context"]       * context_match(outfit_items, context_key)
        + OUTFIT_WEIGHTS["budget"]        * budget_fit(profile, outfit_items)
    )


# ── Candidate generation ──────────────────────────────────────────────────────

def _candidates_by_category(user_id, interactions_by_user, items, top_k):
    """
    Return {category: [top_k items]} for this user, using the same tiered
    candidate logic as the item recommender (unseen → liked → ignored).
    """
    user_interactions = interactions_by_user.get(user_id, [])
    profile = build_user_profile(user_interactions, items)
    if profile is None:
        return {}, None

    by_event      = {i["item_id"]: i["event"] for i in user_interactions}
    seen_ids      = set(by_event)
    liked_ids     = {iid for iid, ev in by_event.items() if ev == "like"}
    ignored_ids   = {iid for iid, ev in by_event.items() if ev == "ignored"}

    unseen  = [it for iid, it in items.items() if iid not in seen_ids]
    liked   = [it for iid, it in items.items() if iid in liked_ids]
    ignored = [it for iid, it in items.items() if iid in ignored_ids]

    # Single ranked list across tiers
    ranked = _score_pool(unseen, profile)
    if len(ranked) < top_k:
        ranked += _score_pool(liked, profile)
    if len(ranked) < top_k:
        ranked += _score_pool(ignored, profile)
    ranked.sort(key=lambda x: x[0], reverse=True)

    by_cat = {}
    for _, item in ranked:
        cat = item["category"]
        if cat not in by_cat:
            by_cat[cat] = []
        if len(by_cat[cat]) < top_k:
            by_cat[cat].append(item)

    return by_cat, profile


def generate_outfits(user_id, interactions_by_user, items, context_key, top_k=6):
    """
    Build all outfit combinations from the top-k candidates per category for
    the given context template.  Returns (scored_outfits, profile) where
    scored_outfits is sorted descending: [(total_score, items, sub_scores), ...].
    """
    by_cat, profile = _candidates_by_category(
        user_id, interactions_by_user, items, top_k
    )
    if profile is None:
        return [], None

    template = OUTFIT_TEMPLATES[CONTEXTS[context_key]["template"]]

    # Verify we have candidates for every required category
    pools = []
    for cat in template:
        pool = by_cat.get(cat, [])
        if not pool:
            return [], profile
        pools.append(pool)

    scored = []
    for combo in cartesian_product(*pools):
        outfit_items = list(combo)
        total = score_outfit(profile, outfit_items, context_key)
        sub = {
            "user_pref":    user_preference_match(profile, outfit_items),
            "compatibility": outfit_compatibility(outfit_items),
            "style_compat": style_compatibility(outfit_items),
            "color_compat": color_compatibility(outfit_items),
            "context":      context_match(outfit_items, context_key),
            "budget":       budget_fit(profile, outfit_items),
        }
        scored.append((total, outfit_items, sub))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored, profile


# ── Outfit explanation ────────────────────────────────────────────────────────

def explain_outfit(outfit_items, sub_scores, profile, context_key):
    reasons = []

    # Style preferences
    top_styles = sorted(profile["style_pref"].items(), key=lambda x: -x[1])[:2]
    outfit_styles = {item["style"] for item in outfit_items}
    matching_styles = [s for s, _ in top_styles if s in outfit_styles]
    if matching_styles:
        reasons.append(f"Matches your preferred styles: {', '.join(matching_styles)}")

    # Color preferences
    top_colors = sorted(profile["color_pref"].items(), key=lambda x: -x[1])[:2]
    outfit_colors = {item["color"] for item in outfit_items}
    matching_colors = [c for c, _ in top_colors if c in outfit_colors]
    if matching_colors:
        reasons.append(f"Uses your preferred colors: {', '.join(matching_colors)}")

    # Style compatibility
    sc = sub_scores["style_compat"]
    if sc == 1.0:
        reasons.append("All pieces are style-compatible with each other")
    elif sc >= 0.5:
        reasons.append("Mostly compatible styles — intentional mixed-style look")
    else:
        reasons.append("Style contrast across pieces (experimental pairing)")

    # Color palette
    cc = sub_scores["color_compat"]
    if cc >= 0.85:
        reasons.append("Clean, cohesive color palette")
    elif cc >= 0.40:
        reasons.append("Bold accent color adds a pop against neutrals")
    else:
        reasons.append("High-contrast color combination (bold choice)")

    # Context fit
    cm = sub_scores["context"]
    label = CONTEXTS[context_key]["label"]
    if cm >= 0.75:
        reasons.append(f"Well suited for: {label}")
    elif cm >= 0.5:
        reasons.append(f"Broadly appropriate for: {label}")
    else:
        reasons.append(f"Style pushes boundaries for: {label}")

    # Budget
    total_price = sum(item["price"] for item in outfit_items)
    if sub_scores["budget"] >= 1.0:
        reasons.append(f"Total ${total_price} fits within your budget")
    else:
        reasons.append(f"Total ${total_price} — slightly above your usual range")

    return reasons


# ── Display ───────────────────────────────────────────────────────────────────

def print_outfit_recommendations(user_id, context_key, scored_outfits, profile, top_n=3):
    label = CONTEXTS[context_key]["label"]
    print(f"\n{'='*64}")
    print(f"  Outfits for {user_id}  —  {label}")
    print(f"{'='*64}")

    if not scored_outfits:
        print("  (not enough item candidates to build outfits for this context)")
        return

    for rank, (total, outfit_items, sub) in enumerate(scored_outfits[:top_n], 1):
        total_price = sum(item["price"] for item in outfit_items)
        print(f"\n  Outfit {rank}  [score: {total:.3f}]")
        print(f"  {'─'*54}")
        for item in outfit_items:
            print(f"  • {item['name']:<38}  {item['category']:<12}  ${item['price']:.0f}")
        print(f"  {'─'*54}")
        print(f"  Outfit total: ${total_price}")
        print()
        print("  Why this outfit:")
        for reason in explain_outfit(outfit_items, sub, profile, context_key):
            print(f"    ✓ {reason}")
        print(
            f"\n  Sub-scores:  "
            f"user_pref={sub['user_pref']:.2f}  "
            f"compatibility={sub['compatibility']:.2f}  "
            f"context={sub['context']:.2f}  "
            f"budget={sub['budget']:.2f}"
        )


# ── Main demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    items                = load_items()
    interactions_by_user = load_interactions()

    demo = [
        ("user_001", "casual_day"),    # Minimalist Neutral
        ("user_008", "casual_day"),    # Streetwear
        ("user_015", "smart_casual"),  # Smart Casual / Preppy
        ("user_022", "cold_weather"),  # Casual with cold weather needs
        ("user_029", "casual_day"),    # Bold / Expressive
    ]

    for user_id, context_key in demo:
        scored_outfits, profile = generate_outfits(
            user_id, interactions_by_user, items, context_key, top_k=6
        )
        print_outfit_recommendations(user_id, context_key, scored_outfits, profile, top_n=3)
