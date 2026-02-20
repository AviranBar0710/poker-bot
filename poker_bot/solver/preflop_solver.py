"""Preflop strategy solver with SQLite GTO database and legacy fallback.

Resolution order:
  1. SQLite database (preflop.db) — precise GTO frequencies from commercial solvers
  2. Legacy JSON (preflop_strategies.json) — generated heuristic strategies
  3. Binary range heuristic — in/out range membership with fabricated frequencies

The SQLite backend is the target for Phase 8a. Once populated with real solver
data, it provides ~95% GTO accuracy for preflop decisions. The legacy tiers
remain as fallbacks for spots not yet imported.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from poker_bot.solver.board_bucketing import bucket_stack
from poker_bot.solver.data_structures import (
    ActionFrequency,
    SolverResult,
    SpotKey,
    StrategyNode,
)
from poker_bot.solver.preflop_db import PreflopDB, nearest_stack_bucket
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

logger = logging.getLogger("poker_bot.solver.preflop")

_JSON_DATA_PATH = Path(__file__).parent / "data" / "preflop_strategies.json"

# Regex to extract raise amount from action strings like "raise_2.5"
_RAISE_AMOUNT_RE = re.compile(r"^raise_(\d+\.?\d*)$")


def _count_raises(history: list[PriorAction]) -> int:
    return sum(1 for a in history if a.action in (Action.RAISE, Action.ALL_IN))


def _has_limps(history: list[PriorAction]) -> bool:
    return any(a.action in (Action.CALL, Action.LIMP) for a in history)


def _action_sequence(history: list[PriorAction]) -> str:
    """Determine the action sequence bucket from history."""
    raises = _count_raises(history)
    limps = _has_limps(history)

    if raises >= 3:
        return "vs_4bet"
    if raises >= 2:
        return "vs_3bet"
    if raises >= 1:
        return "vs_raise"
    if limps:
        return "vs_limp"
    return "open"


def _parse_db_action(action_str: str) -> tuple[str, float]:
    """Parse a DB action string into (base_action, amount).

    Examples:
        "fold"        → ("fold", 0.0)
        "limp"        → ("limp", 1.0)
        "call"        → ("call", 0.0)
        "raise_2.5"   → ("raise", 2.5)
        "raise_3.0"   → ("raise", 3.0)
        "raise_all_in"→ ("all_in", 0.0)
        "raise"       → ("raise", 2.5)  # default sizing
    """
    if action_str == "raise_all_in":
        return "all_in", 0.0

    match = _RAISE_AMOUNT_RE.match(action_str)
    if match:
        return "raise", float(match.group(1))

    if action_str == "raise":
        return "raise", 2.5  # default open size

    if action_str == "limp":
        return "limp", 1.0

    # fold, call, check, etc.
    return action_str, 0.0


class PreflopSolver:
    """Preflop strategy solver with three-tier resolution."""

    def __init__(
        self,
        data_path: Path | str | None = None,
        db_path: Path | str | None = None,
    ) -> None:
        # Tier 1: SQLite database (precise GTO data)
        self._db: PreflopDB | None = None
        try:
            self._db = PreflopDB(db_path=db_path)
            count = self._db.row_count()
            if count > 0:
                logger.info("Preflop DB loaded: %d rows", count)
            else:
                logger.debug("Preflop DB exists but is empty")
        except Exception:
            logger.debug("No preflop DB available, using legacy only")
            self._db = None

        # Tier 2: Legacy JSON data
        self._json_data: dict = {}
        json_path = Path(data_path) if data_path else _JSON_DATA_PATH
        if json_path.exists():
            with open(json_path) as f:
                self._json_data = json.load(f)

    def _lookup_db(
        self,
        hand_str: str,
        position: str,
        action_seq: str,
        stack_bb: float,
    ) -> StrategyNode | None:
        """Try the SQLite database for a precise GTO strategy."""
        if self._db is None:
            return None

        stack_bucket = nearest_stack_bucket(stack_bb)
        rows = self._db.lookup(position, action_seq, stack_bucket, hand_str)

        if not rows:
            return None

        actions = []
        for action_str, frequency, ev in rows:
            base_action, amount = _parse_db_action(action_str)
            actions.append(ActionFrequency(base_action, frequency, amount, ev))

        return StrategyNode(actions=actions)

    def _lookup_json(
        self,
        hand_str: str,
        position: str,
        action_seq: str,
    ) -> StrategyNode | None:
        """Try the legacy JSON data."""
        pos_data = self._json_data.get(position, {})
        seq_data = pos_data.get(action_seq, {})
        actions_data = seq_data.get(hand_str)
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

    # Keep the old lookup() interface for backward compatibility with tests
    def lookup(
        self,
        hand: str,
        position: str,
        action_seq: str,
    ) -> StrategyNode | None:
        """Look up a pre-computed strategy (JSON only, legacy interface)."""
        return self._lookup_json(hand, position, action_seq)

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

        Resolution order:
          1. SQLite DB (confidence=0.95) — real solver data
          2. Legacy JSON (confidence=0.85) — generated heuristics
          3. Range heuristic (confidence=0.4) — binary in/out

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

        spot_key = SpotKey(
            street="preflop",
            position=pos_str,
            action_sequence=action_seq,
            stack_bucket=nearest_stack_bucket(stack_bb),
        )

        # Tier 1: SQLite database (precise GTO)
        node = self._lookup_db(hand_str, pos_str, action_seq, stack_bb)
        if node is not None:
            if survival_premium < 0.95:
                node = self._apply_icm(node, survival_premium)
            return SolverResult(
                strategy=node,
                source="preflop_db",
                confidence=0.95,
                ev=node.weighted_ev,
                spot_key=spot_key,
            )

        # Tier 2: Legacy JSON lookup
        node = self._lookup_json(hand_str, pos_str, action_seq)
        if node is not None:
            if survival_premium < 0.95:
                node = self._apply_icm(node, survival_premium)
            return SolverResult(
                strategy=node,
                source="preflop_lookup",
                confidence=0.85,
                ev=node.weighted_ev,
                spot_key=spot_key,
            )

        # Tier 3: Binary range heuristic
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
            actions = [
                ActionFrequency("fold", 0.95, 0.0, 0.0),
                ActionFrequency("raise", 0.05, 2.5, -0.5),
            ]

        return StrategyNode(actions=actions)

    @staticmethod
    def _apply_icm(node: StrategyNode, survival_premium: float) -> StrategyNode:
        """Adjust mixed strategy for ICM pressure."""
        icm_factor = 1.0 - survival_premium

        adjusted = []
        for af in node.actions:
            if af.action == "fold":
                new_freq = af.frequency + icm_factor * 0.3
                adjusted.append(ActionFrequency(
                    af.action, new_freq, af.amount, af.ev,
                ))
            else:
                new_freq = max(0.0, af.frequency * (1.0 - icm_factor * 0.4))
                adjusted.append(ActionFrequency(
                    af.action, new_freq, af.amount, af.ev,
                ))

        return StrategyNode(actions=adjusted).normalized()
