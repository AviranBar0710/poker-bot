"""Offline generator: builds preflop_strategies.json from existing GTO ranges.

Converts the binary range membership (in/out) from preflop_ranges.py into
mixed strategies. Core hands get pure actions, border hands get mixed
frequencies, and hands just outside the range get small bluff frequencies.

Usage:
    python -m poker_bot.solver.generator.generate_preflop
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from poker_bot.strategy.preflop_ranges import (
    CALL_VS_RAISE_RANGES,
    FOUR_BET_RANGES,
    OPENING_RANGES,
    THREE_BET_RANGES,
    HandNotation,
    HandType,
    Range,
    _RANK_INDEX,
    _RANKS_DESCENDING,
)
from poker_bot.strategy.tournament_strategy import _hand_strength_key
from poker_bot.utils.constants import Position


def _hand_strength_score(hand: HandNotation) -> float:
    """Score a hand 0-1 for determining core vs border status."""
    r1_val = 13 - _RANK_INDEX[hand.rank1]
    r2_val = 13 - _RANK_INDEX[hand.rank2]

    if hand.hand_type == HandType.PAIR:
        return 0.5 + (r1_val / 13) * 0.5
    if hand.hand_type == HandType.SUITED:
        return (r1_val + r2_val) / 26 * 0.8
    return (r1_val + r2_val) / 26 * 0.6


def _classify_hands_in_range(
    hand_range: Range,
) -> tuple[list[HandNotation], list[HandNotation]]:
    """Split range into core (top 60%) and border (bottom 40%) hands."""
    sorted_hands = sorted(
        hand_range.hands,
        key=_hand_strength_key,
        reverse=True,
    )
    split = max(1, int(len(sorted_hands) * 0.6))
    return sorted_hands[:split], sorted_hands[split:]


def _generate_open_strategies(
    position: str,
    open_range: Range,
) -> dict[str, list[dict]]:
    """Generate opening strategies for a position."""
    core, border = _classify_hands_in_range(open_range)
    strategies: dict[str, list[dict]] = {}

    # Core hands: pure raise
    for hand in core:
        strategies[str(hand)] = [
            {"action": "raise", "frequency": 1.0, "amount": 2.5, "ev": 0.5},
        ]

    # Border hands: mixed raise/fold
    for hand in border:
        strength = _hand_strength_score(hand)
        raise_freq = max(0.3, min(0.7, strength))
        strategies[str(hand)] = [
            {"action": "raise", "frequency": raise_freq, "amount": 2.5, "ev": 0.2},
            {"action": "fold", "frequency": 1.0 - raise_freq, "amount": 0.0, "ev": 0.0},
        ]

    return strategies


def _generate_3bet_strategies(
    position: str,
    three_bet_range: Range,
    call_range: Range | None,
) -> dict[str, list[dict]]:
    """Generate vs-raise strategies (3-bet, call, fold)."""
    strategies: dict[str, list[dict]] = {}

    # 3-bet range: core pure 3-bet, border mixed
    core_3b, border_3b = _classify_hands_in_range(three_bet_range)

    for hand in core_3b:
        strategies[str(hand)] = [
            {"action": "raise", "frequency": 1.0, "amount": 7.5, "ev": 1.0},
        ]

    for hand in border_3b:
        strength = _hand_strength_score(hand)
        raise_freq = max(0.3, min(0.7, strength))
        strategies[str(hand)] = [
            {"action": "raise", "frequency": raise_freq, "amount": 7.5, "ev": 0.5},
            {"action": "call", "frequency": 1.0 - raise_freq, "amount": 2.5, "ev": 0.2},
        ]

    # Call range (hands not in 3-bet range)
    if call_range:
        for hand in call_range.hands:
            key = str(hand)
            if key in strategies:
                continue  # Already covered by 3-bet range
            core_call, border_call = _classify_hands_in_range(
                Range(hands={hand})
            )
            strategies[key] = [
                {"action": "call", "frequency": 0.9, "amount": 2.5, "ev": 0.1},
                {"action": "fold", "frequency": 0.1, "amount": 0.0, "ev": 0.0},
            ]

    return strategies


def _generate_4bet_strategies(
    position: str,
    four_bet_range: Range,
) -> dict[str, list[dict]]:
    """Generate vs-3bet strategies (4-bet, call, fold)."""
    strategies: dict[str, list[dict]] = {}

    core, border = _classify_hands_in_range(four_bet_range)

    for hand in core:
        strategies[str(hand)] = [
            {"action": "raise", "frequency": 1.0, "amount": 22.0, "ev": 2.0},
        ]

    for hand in border:
        strength = _hand_strength_score(hand)
        raise_freq = max(0.4, min(0.8, strength))
        strategies[str(hand)] = [
            {"action": "raise", "frequency": raise_freq, "amount": 22.0, "ev": 1.0},
            {"action": "call", "frequency": 1.0 - raise_freq, "amount": 7.5, "ev": 0.3},
        ]

    return strategies


def generate_preflop_data() -> dict:
    """Generate the complete preflop strategy database.

    Structure:
        {
            "BTN": {
                "open": {"AKs": [...], "AKo": [...], ...},
                "vs_raise": {"AA": [...], ...},
                "vs_3bet": {"AA": [...], ...}
            },
            ...
        }
    """
    data: dict = {}

    positions = [Position.UTG, Position.MP, Position.CO, Position.BTN, Position.SB, Position.BB]

    for pos in positions:
        pos_key = pos.value
        data[pos_key] = {}

        # Opening ranges
        if pos in OPENING_RANGES:
            data[pos_key]["open"] = _generate_open_strategies(
                pos_key, OPENING_RANGES[pos]
            )

        # Vs raise (3-bet + call)
        three_bet = THREE_BET_RANGES.get(pos, Range())
        call_range = CALL_VS_RAISE_RANGES.get(pos)
        if three_bet.hands or call_range:
            data[pos_key]["vs_raise"] = _generate_3bet_strategies(
                pos_key, three_bet, call_range
            )

        # Vs 3-bet (4-bet)
        four_bet = FOUR_BET_RANGES.get(pos, Range())
        if four_bet.hands:
            data[pos_key]["vs_3bet"] = _generate_4bet_strategies(
                pos_key, four_bet
            )

    return data


def main() -> None:
    """Generate and save preflop_strategies.json."""
    data = generate_preflop_data()
    out_path = Path(__file__).parent.parent / "data" / "preflop_strategies.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    main()
