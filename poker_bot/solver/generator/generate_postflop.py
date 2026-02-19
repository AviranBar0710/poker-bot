"""Offline generator: builds postflop_strategies.json from heuristic GTO rules.

Encodes standard GTO postflop strategies including:
- C-bet frequencies by board texture
- Check-raise frequencies
- Value/bluff bet ratios by sizing
- Hand category responses to various board types

Usage:
    python -m poker_bot.solver.generator.generate_postflop
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Board texture buckets
BOARD_BUCKETS = [
    "dry_high_rainbow", "dry_low_rainbow", "dry_medium",
    "wet_connected", "wet_two_tone",
    "monotone_high", "monotone_low",
    "paired_high", "paired_low",
    "broadway_heavy", "connected_low", "dynamic",
]

# Hand categories
HAND_CATEGORIES = [
    "nuts", "strong_made", "medium_made", "weak_made",
    "strong_draw", "medium_draw", "weak_draw", "air",
]

SPR_BUCKETS = ["low", "medium", "high"]


def _strategy(actions: list[tuple[str, float, float, float]]) -> list[dict]:
    """Build strategy list from (action, freq, amount_frac, ev) tuples."""
    return [
        {"action": a, "frequency": f, "amount": amt, "ev": ev}
        for a, f, amt, ev in actions
    ]


def _generate_board_strategies(board_bucket: str) -> dict:
    """Generate hand-category strategies for a board texture."""
    # Determine board character
    is_dry = "dry" in board_bucket or "paired" in board_bucket
    is_wet = "wet" in board_bucket or "connected" in board_bucket or "dynamic" in board_bucket
    is_monotone = "monotone" in board_bucket
    is_high = "high" in board_bucket or "broadway" in board_bucket

    # Base c-bet sizing fraction
    if is_dry:
        cbet_size = 0.33
    elif is_monotone:
        cbet_size = 0.75
    elif is_wet:
        cbet_size = 0.66
    else:
        cbet_size = 0.50

    strategies: dict = {}

    for category in HAND_CATEGORIES:
        cat_strats: dict = {}
        for spr_bucket in SPR_BUCKETS:
            cat_strats[spr_bucket] = _category_strategy(
                category, board_bucket, spr_bucket,
                is_dry, is_wet, is_monotone, is_high, cbet_size,
            )
        strategies[category] = cat_strats

    return strategies


def _category_strategy(
    category: str,
    board_bucket: str,
    spr_bucket: str,
    is_dry: bool,
    is_wet: bool,
    is_monotone: bool,
    is_high: bool,
    cbet_size: float,
) -> list[dict]:
    """Generate strategy for a specific hand category + texture + SPR."""
    low_spr = spr_bucket == "low"
    high_spr = spr_bucket == "high"

    if category == "nuts":
        if low_spr:
            return _strategy([("raise", 0.90, cbet_size * 1.5, 3.0), ("call", 0.10, 0.0, 2.0)])
        if is_wet:
            return _strategy([("raise", 0.85, cbet_size, 2.5), ("call", 0.10, 0.0, 1.5), ("check", 0.05, 0.0, 1.0)])
        return _strategy([("raise", 0.80, cbet_size, 2.0), ("call", 0.12, 0.0, 1.5), ("check", 0.08, 0.0, 1.0)])

    if category == "strong_made":
        if low_spr:
            return _strategy([("raise", 0.80, cbet_size * 1.2, 2.0), ("call", 0.15, 0.0, 1.0), ("fold", 0.05, 0.0, 0.0)])
        if is_wet:
            return _strategy([("raise", 0.70, cbet_size, 1.5), ("call", 0.20, 0.0, 0.8), ("check", 0.10, 0.0, 0.5)])
        return _strategy([("raise", 0.65, cbet_size, 1.2), ("call", 0.22, 0.0, 0.7), ("check", 0.13, 0.0, 0.4)])

    if category == "medium_made":
        if low_spr:
            return _strategy([("raise", 0.40, cbet_size, 0.5), ("call", 0.35, 0.0, 0.3), ("check", 0.15, 0.0, 0.1), ("fold", 0.10, 0.0, 0.0)])
        if is_monotone:
            return _strategy([("check", 0.40, 0.0, 0.1), ("call", 0.30, 0.0, 0.2), ("raise", 0.20, cbet_size, 0.3), ("fold", 0.10, 0.0, 0.0)])
        return _strategy([("raise", 0.35, cbet_size * 0.8, 0.4), ("call", 0.35, 0.0, 0.3), ("check", 0.25, 0.0, 0.1), ("fold", 0.05, 0.0, 0.0)])

    if category == "weak_made":
        if is_wet or is_monotone:
            return _strategy([("check", 0.45, 0.0, 0.0), ("fold", 0.30, 0.0, 0.0), ("call", 0.20, 0.0, -0.1), ("raise", 0.05, cbet_size, -0.2)])
        return _strategy([("check", 0.55, 0.0, 0.0), ("call", 0.25, 0.0, -0.1), ("fold", 0.15, 0.0, 0.0), ("raise", 0.05, cbet_size * 0.5, -0.1)])

    if category == "strong_draw":
        if low_spr:
            return _strategy([("raise", 0.60, cbet_size * 1.5, 1.0), ("call", 0.30, 0.0, 0.5), ("fold", 0.10, 0.0, 0.0)])
        return _strategy([("raise", 0.50, cbet_size, 0.8), ("call", 0.35, 0.0, 0.4), ("check", 0.15, 0.0, 0.1)])

    if category == "medium_draw":
        if high_spr:
            return _strategy([("call", 0.40, 0.0, 0.2), ("check", 0.35, 0.0, 0.0), ("raise", 0.25, cbet_size, 0.3)])
        return _strategy([("check", 0.40, 0.0, 0.0), ("call", 0.35, 0.0, 0.1), ("raise", 0.20, cbet_size * 0.8, 0.1), ("fold", 0.05, 0.0, 0.0)])

    if category == "weak_draw":
        return _strategy([("check", 0.50, 0.0, 0.0), ("fold", 0.35, 0.0, 0.0), ("call", 0.15, 0.0, -0.1)])

    # air
    if is_dry and not is_high:
        # Bluff more on dry low boards
        return _strategy([("fold", 0.55, 0.0, 0.0), ("raise", 0.20, cbet_size, -0.3), ("check", 0.25, 0.0, 0.0)])
    if is_wet or is_monotone:
        return _strategy([("fold", 0.70, 0.0, 0.0), ("check", 0.20, 0.0, 0.0), ("raise", 0.10, cbet_size, -0.5)])
    return _strategy([("fold", 0.60, 0.0, 0.0), ("check", 0.25, 0.0, 0.0), ("raise", 0.15, cbet_size, -0.4)])


def generate_postflop_data() -> dict:
    """Generate the complete postflop strategy database."""
    data: dict = {}
    for bucket in BOARD_BUCKETS:
        data[bucket] = _generate_board_strategies(bucket)
    return data


def main() -> None:
    """Generate and save postflop_strategies.json."""
    data = generate_postflop_data()
    out_path = Path(__file__).parent.parent / "data" / "postflop_strategies.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    main()
