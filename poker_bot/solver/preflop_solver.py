"""Pre-computed preflop strategy lookup.

Loads preflop_strategies.json and provides O(1) strategy lookup
by hand notation, position, action sequence, and stack bucket.
Falls back to heuristic when no pre-computed strategy exists.
"""

from __future__ import annotations

import json
from pathlib import Path

from poker_bot.solver.board_bucketing import bucket_stack
from poker_bot.solver.data_structures import (
    ActionFrequency,
    SolverResult,
    SpotKey,
    StrategyNode,
)
from poker_bot.strategy.decision_maker import PriorAction, _hand_to_notation
from poker_bot.strategy.preflop_ranges import (
    CALL_VS_RAISE_RANGES,
    FOUR_BET_RANGES,
    OPENING_RANGES,
    THREE_BET_RANGES,
    HandNotation,
    Range,
    _RANK_INDEX,
)
from poker_bot.utils.card import Card
from poker_bot.utils.constants import Action, Position

_DATA_PATH = Path(__file__).parent / "data" / "preflop_strategies.json"


def _count_raises(history: list[PriorAction]) -> int:
    return sum(1 for a in history if a.action in (Action.RAISE, Action.ALL_IN))


def _action_sequence(history: list[PriorAction]) -> str:
    """Determine the action sequence bucket from history."""
    raises = _count_raises(history)
    if raises >= 3:
        return "vs_4bet"
    if raises >= 2:
        return "vs_3bet"
    if raises >= 1:
        return "vs_raise"
    return "open"


class PreflopSolver:
    """Pre-computed preflop strategy lookup engine."""

    def __init__(self, data_path: Path | str | None = None) -> None:
        self._data: dict = {}
        path = Path(data_path) if data_path else _DATA_PATH
        if path.exists():
            with open(path) as f:
                self._data = json.load(f)

    def lookup(
        self,
        hand: str,
        position: str,
        action_seq: str,
    ) -> StrategyNode | None:
        """Look up a pre-computed strategy.

        Args:
            hand: Hand notation string (e.g. "AKs").
            position: Position string (e.g. "BTN").
            action_seq: Action sequence (e.g. "open", "vs_raise").

        Returns:
            StrategyNode if found, None otherwise.
        """
        pos_data = self._data.get(position, {})
        seq_data = pos_data.get(action_seq, {})
        actions_data = seq_data.get(hand)
        if actions_data is None:
            return None
        actions = [
            ActionFrequency(
                action=a["action"],
                frequency=a["frequency"],
                amount=a.get("amount", 0.0),
                ev=a.get("ev", 0.0),
            )
            for a in actions_data
        ]
        return StrategyNode(actions=actions)

    def get_strategy(
        self,
        card1: Card,
        card2: Card,
        position: Position,
        action_history: list[PriorAction],
        stack_bb: float,
        survival_premium: float = 1.0,
    ) -> SolverResult:
        """Get a complete strategy for a preflop spot.

        Tries pre-computed lookup first, then falls back to heuristic
        range-based strategy.

        Args:
            card1: First hole card.
            card2: Second hole card.
            position: Hero's position.
            action_history: Actions so far in the hand.
            stack_bb: Hero's stack in big blinds.
            survival_premium: ICM survival premium [0.3, 1.0].

        Returns:
            SolverResult with strategy, source, and confidence.
        """
        hand = _hand_to_notation(card1, card2)
        hand_str = str(hand)
        pos_str = position.value
        action_seq = _action_sequence(action_history)
        stack_bucket = bucket_stack(stack_bb)

        spot_key = SpotKey(
            street="preflop",
            position=pos_str,
            action_sequence=action_seq,
            stack_bucket=stack_bucket,
        )

        # Try pre-computed lookup
        node = self.lookup(hand_str, pos_str, action_seq)

        if node is not None:
            # Apply ICM adjustment if needed
            if survival_premium < 0.95:
                node = self._apply_icm(node, survival_premium)

            return SolverResult(
                strategy=node,
                source="preflop_lookup",
                confidence=0.85,
                ev=node.weighted_ev,
                spot_key=spot_key,
            )

        # Fallback: check if hand is in the relevant binary range
        node = self._heuristic_fallback(
            card1, card2, position, action_seq, survival_premium,
        )
        return SolverResult(
            strategy=node,
            source="heuristic",
            confidence=0.4,
            ev=node.weighted_ev,
            spot_key=spot_key,
        )

    def _heuristic_fallback(
        self,
        card1: Card,
        card2: Card,
        position: Position,
        action_seq: str,
        survival_premium: float,
    ) -> StrategyNode:
        """Build a strategy from the binary range system."""
        range_map = {
            "open": OPENING_RANGES,
            "vs_raise": THREE_BET_RANGES,
            "vs_3bet": FOUR_BET_RANGES,
        }
        target_range = range_map.get(action_seq, {}).get(position, Range())

        if target_range.hands and target_range.contains(card1, card2):
            # Hand is in range — default to action
            action = "raise" if action_seq != "vs_raise" else "raise"
            amount = {"open": 2.5, "vs_raise": 7.5, "vs_3bet": 22.0}.get(
                action_seq, 2.5
            )
            freq = max(0.5, survival_premium)
            actions = [
                ActionFrequency(action, freq, amount, 0.3),
                ActionFrequency("fold", 1.0 - freq, 0.0, 0.0),
            ]
        else:
            # Not in range — mostly fold, tiny bluff frequency
            actions = [
                ActionFrequency("fold", 0.95, 0.0, 0.0),
                ActionFrequency("raise", 0.05, 2.5, -0.5),
            ]

        return StrategyNode(actions=actions)

    @staticmethod
    def _apply_icm(node: StrategyNode, survival_premium: float) -> StrategyNode:
        """Adjust mixed strategy for ICM pressure.

        Increases fold frequency, decreases aggressive actions,
        then renormalizes.
        """
        icm_factor = 1.0 - survival_premium  # Higher = more pressure

        adjusted = []
        for af in node.actions:
            if af.action == "fold":
                # Increase fold frequency
                new_freq = af.frequency + icm_factor * 0.3
                adjusted.append(ActionFrequency(
                    af.action, new_freq, af.amount, af.ev,
                ))
            else:
                # Decrease aggressive action frequency
                new_freq = max(0.0, af.frequency * (1.0 - icm_factor * 0.4))
                adjusted.append(ActionFrequency(
                    af.action, new_freq, af.amount, af.ev,
                ))

        return StrategyNode(actions=adjusted).normalized()
