"""
Generate 50 synthetic users by sampling from archetypes with small random variations.

Each archetype produces ~6-7 users. Variations applied per user:
  - Budget range shifted ±20%
  - Weights slightly jittered (then renormalized)
  - Occasionally one extra color preference added
"""

import json
import random
from pathlib import Path

SEED = 42
NUM_USERS = 50
EXTRA_COLORS = ["olive", "burgundy", "tan", "cream", "charcoal", "forest green"]

random.seed(SEED)


def jitter_weights(archetype: dict, noise: float = 0.05) -> dict:
    """Add small noise to weights and renormalize so they still sum to 1."""
    keys = ["style_weight", "color_weight", "price_weight", "category_weight"]
    raw = {k: max(0.05, archetype[k] + random.uniform(-noise, noise)) for k in keys}
    total = sum(raw.values())
    return {k: round(v / total, 4) for k, v in raw.items()}


def jitter_budget(budget_range: list, noise_pct: float = 0.20) -> list:
    low, high = budget_range
    jitter = random.uniform(-noise_pct, noise_pct)
    new_low = max(5, round(low * (1 + jitter)))
    new_high = round(high * (1 + jitter))
    return [new_low, new_high]


def maybe_add_color(colors: list) -> list:
    """With 30% probability, append a random extra color."""
    if random.random() < 0.3:
        extra = random.choice([c for c in EXTRA_COLORS if c not in colors])
        return colors + [extra]
    return colors


def sample_users(archetypes: list, total: int) -> list:
    users = []
    user_id = 1

    # Distribute users across archetypes as evenly as possible
    n = len(archetypes)
    base, remainder = divmod(total, n)
    counts = [base + (1 if i < remainder else 0) for i in range(n)]

    for archetype, count in zip(archetypes, counts):
        for _ in range(count):
            weights = jitter_weights(archetype)
            user = {
                "user_id": f"user_{user_id:03d}",
                "archetype": archetype["archetype_name"],
                "preferred_styles": list(archetype["preferred_styles"]),
                "preferred_colors": maybe_add_color(list(archetype["preferred_colors"])),
                "disliked_colors": list(archetype["disliked_colors"]),
                "preferred_categories": list(archetype["preferred_categories"]),
                "budget_range": jitter_budget(archetype["budget_range"]),
                **weights,
            }
            users.append(user)
            user_id += 1

    return users


def main():
    archetypes_path = Path("archetypes.json")
    output_path = Path("users.json")

    with archetypes_path.open() as f:
        archetypes = json.load(f)

    users = sample_users(archetypes, NUM_USERS)

    with output_path.open("w") as f:
        json.dump(users, f, indent=2)

    print(f"Generated {len(users)} users → {output_path}")

    # Quick summary
    from collections import Counter
    counts = Counter(u["archetype"] for u in users)
    for archetype, count in counts.most_common():
        print(f"  {count:2d}  {archetype}")


if __name__ == "__main__":
    main()
