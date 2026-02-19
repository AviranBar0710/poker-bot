"""Postflop strategy solver with pre-computed lookup and Monte Carlo fallback.

Combines pre-computed heuristic GTO strategies for common board
textures with real-time equity-based fallback for unusual spots.
"""

from __future__ import annotations

import json
from pathlib import Path

from poker_bot.core.equity_calculator import EquityCalculator
from poker_bot.solver.bet_sizing import BetSizingTree
from poker_bot.solver.board_bucketing import bucket_board, bucket_spr, bucket_stack
from poker_bot.solver.data_structures import (
    ActionFrequency,
    SolverResult,
    SpotKey,
    StrategyNode,
)
from poker_bot.solver.range_estimator import RangeEstimator
from poker_bot.strategy.decision_maker import (
    BoardTexture,
    PriorAction,
    analyze_board,
)
from poker_bot.strategy.preflop_ranges import Range
from poker_bot.interface.opponent_tracker import OpponentStats
from poker_bot.utils.card import Card
from poker_bot.utils.constants import Action, Position, Street

# GTO baseline stats (used when sample size is insufficient)
_GTO_DEFAULTS = {
    "vpip": 22.0,
    "pfr": 18.0,
    "three_bet": 7.0,
    "fold_to_cbet": 50.0,
    "aggression_factor": 2.0,
}

# Minimum sample sizes before exploitative adjustments begin
_MIN_SAMPLES = {
    "vpip": 30,       # hands_seen for VPIP/PFR
    "pfr": 30,
    "three_bet": 50,   # hands_seen for 3-bet
    "fold_to_cbet": 10,  # cbet_faced instances
}

# Full confidence at 3× the minimum threshold
_FULL_CONFIDENCE_MULT = 3

_DATA_PATH = Path(__file__).parent / "data" / "postflop_strategies.json"


# Heuristic strategy tables: hand_category -> board_bucket_type -> strategy
# Board bucket types: "dry", "wet", "monotone", "paired"
def _board_bucket_type(bucket: str) -> str:
    if "monotone" in bucket:
        return "monotone"
    if "paired" in bucket:
        return "paired"
    if "wet" in bucket or "connected" in bucket or "dynamic" in bucket or "broadway" in bucket:
        return "wet"
    return "dry"


# Heuristic strategies by hand category and general texture type
# Format: {hand_category: {texture_type: [(action, freq, sizing_frac)]}}
_HEURISTIC_STRATEGIES: dict[str, dict[str, list[tuple[str, float, float]]]] = {
    "nuts": {
        "dry":     [("raise", 0.85, 0.33), ("call", 0.10, 0.0), ("check", 0.05, 0.0)],
        "wet":     [("raise", 0.90, 0.66), ("call", 0.05, 0.0), ("check", 0.05, 0.0)],
        "monotone": [("raise", 0.80, 0.75), ("call", 0.15, 0.0), ("check", 0.05, 0.0)],
        "paired":  [("raise", 0.85, 0.33), ("call", 0.10, 0.0), ("check", 0.05, 0.0)],
    },
    "strong_made": {
        "dry":     [("raise", 0.70, 0.33), ("call", 0.20, 0.0), ("check", 0.10, 0.0)],
        "wet":     [("raise", 0.75, 0.66), ("call", 0.15, 0.0), ("check", 0.10, 0.0)],
        "monotone": [("raise", 0.65, 0.75), ("call", 0.25, 0.0), ("check", 0.10, 0.0)],
        "paired":  [("raise", 0.70, 0.33), ("call", 0.20, 0.0), ("check", 0.10, 0.0)],
    },
    "medium_made": {
        "dry":     [("raise", 0.35, 0.33), ("call", 0.40, 0.0), ("check", 0.25, 0.0)],
        "wet":     [("raise", 0.40, 0.50), ("call", 0.35, 0.0), ("check", 0.25, 0.0)],
        "monotone": [("raise", 0.25, 0.75), ("call", 0.40, 0.0), ("check", 0.35, 0.0)],
        "paired":  [("raise", 0.35, 0.33), ("call", 0.40, 0.0), ("check", 0.25, 0.0)],
    },
    "weak_made": {
        "dry":     [("check", 0.55, 0.0), ("call", 0.30, 0.0), ("raise", 0.10, 0.33), ("fold", 0.05, 0.0)],
        "wet":     [("check", 0.45, 0.0), ("call", 0.30, 0.0), ("raise", 0.15, 0.50), ("fold", 0.10, 0.0)],
        "monotone": [("check", 0.50, 0.0), ("fold", 0.25, 0.0), ("call", 0.20, 0.0), ("raise", 0.05, 0.75)],
        "paired":  [("check", 0.55, 0.0), ("call", 0.30, 0.0), ("fold", 0.10, 0.0), ("raise", 0.05, 0.33)],
    },
    "strong_draw": {
        "dry":     [("raise", 0.50, 0.66), ("call", 0.35, 0.0), ("check", 0.15, 0.0)],
        "wet":     [("raise", 0.55, 0.66), ("call", 0.30, 0.0), ("check", 0.15, 0.0)],
        "monotone": [("raise", 0.45, 0.75), ("call", 0.35, 0.0), ("check", 0.20, 0.0)],
        "paired":  [("raise", 0.45, 0.50), ("call", 0.35, 0.0), ("check", 0.20, 0.0)],
    },
    "medium_draw": {
        "dry":     [("check", 0.45, 0.0), ("call", 0.35, 0.0), ("raise", 0.20, 0.50)],
        "wet":     [("call", 0.40, 0.0), ("check", 0.35, 0.0), ("raise", 0.25, 0.66)],
        "monotone": [("call", 0.40, 0.0), ("check", 0.40, 0.0), ("raise", 0.20, 0.75)],
        "paired":  [("check", 0.45, 0.0), ("call", 0.35, 0.0), ("raise", 0.20, 0.50)],
    },
    "weak_draw": {
        "dry":     [("check", 0.55, 0.0), ("fold", 0.30, 0.0), ("call", 0.15, 0.0)],
        "wet":     [("check", 0.45, 0.0), ("fold", 0.35, 0.0), ("call", 0.20, 0.0)],
        "monotone": [("fold", 0.45, 0.0), ("check", 0.40, 0.0), ("call", 0.15, 0.0)],
        "paired":  [("check", 0.55, 0.0), ("fold", 0.30, 0.0), ("call", 0.15, 0.0)],
    },
    "air": {
        "dry":     [("fold", 0.60, 0.0), ("check", 0.25, 0.0), ("raise", 0.15, 0.66)],
        "wet":     [("fold", 0.70, 0.0), ("check", 0.20, 0.0), ("raise", 0.10, 0.66)],
        "monotone": [("fold", 0.75, 0.0), ("check", 0.20, 0.0), ("raise", 0.05, 0.75)],
        "paired":  [("fold", 0.60, 0.0), ("check", 0.25, 0.0), ("raise", 0.15, 0.33)],
    },
}


class PostflopSolver:
    """Postflop strategy solver with lookup and Monte Carlo fallback."""

    def __init__(self, data_path: Path | str | None = None) -> None:
        self._data: dict = {}
        path = Path(data_path) if data_path else _DATA_PATH
        if path.exists():
            with open(path) as f:
                self._data = json.load(f)

    # Postflop action order: SB, BB act first; then UTG, MP, CO, BTN last
    _POSTFLOP_ORDER: dict[str, int] = {
        "SB": 0, "BB": 1, "UTG": 2, "MP": 3, "CO": 4, "BTN": 5,
    }

    def get_strategy(
        self,
        hero_cards: list[Card],
        community_cards: list[Card],
        position: Position,
        pot: float,
        hero_stack: float,
        big_blind: float,
        action_history: list[PriorAction],
        hand_strength: float,
        has_draw: bool = False,
        draw_strength: float = 0.0,
        opponent_range: Range | None = None,
        num_opponents: int = 1,
        is_ip: bool = True,
        opponent_stats: OpponentStats | None = None,
    ) -> SolverResult:
        """Get a mixed strategy for a postflop spot.

        Args:
            hero_cards: Hero's hole cards.
            community_cards: Board cards.
            position: Hero's position.
            pot: Current pot size in bb.
            hero_stack: Hero's remaining stack in bb.
            big_blind: Big blind size.
            action_history: Prior actions.
            hand_strength: Hero's hand strength [0, 1].
            has_draw: Whether hero has a significant draw.
            draw_strength: Draw strength [0, 1].
            opponent_range: Optional explicit opponent range.
            num_opponents: Number of opponents still in the hand (default 1).
            is_ip: Whether hero is in position (acts last) (default True).
            opponent_stats: Optional opponent statistics for exploitative adjustments.

        Returns:
            SolverResult with mixed strategy.
        """
        texture = analyze_board(community_cards)
        board_bucket = bucket_board(texture)
        spr = hero_stack / pot if pot > 0 else float("inf")
        spr_bucket = bucket_spr(spr)
        stack_bucket = bucket_stack(hero_stack)

        hand_category = RangeEstimator.categorize_hand(
            hand_strength, has_draw, draw_strength,
        )

        street = self._detect_street(community_cards)

        spot_key = SpotKey(
            street=street,
            position=position.value,
            action_sequence=self._action_seq(action_history),
            stack_bucket=stack_bucket,
            board_bucket=board_bucket,
            spr_bucket=spr_bucket,
            hand_category=hand_category,
        )

        # Try pre-computed lookup
        node = self._lookup(board_bucket, position.value, hand_category, spr_bucket)
        if node is not None:
            # Convert sizing fractions to actual amounts
            node = self._resolve_amounts(node, pot, hero_stack, big_blind, board_bucket)
            node = self._apply_position_adjustment(node, is_ip, hand_category)
            if num_opponents >= 2:
                node = self._apply_multiway_adjustment(node, num_opponents, hand_category)
            if opponent_stats is not None:
                node = self._apply_exploit_adjustment(node, opponent_stats, hand_category)
            return SolverResult(
                strategy=node,
                source="postflop_lookup",
                confidence=0.75,
                ev=node.weighted_ev,
                spot_key=spot_key,
            )

        # Heuristic fallback based on hand category and texture
        node = self._heuristic_strategy(
            hand_category, board_bucket, pot, hero_stack, big_blind, spr,
        )
        confidence = 0.6

        # Monte Carlo refinement for close decisions
        if opponent_range and 0.30 <= hand_strength <= 0.75:
            try:
                mc_equity = self._monte_carlo_equity(
                    hero_cards, community_cards, opponent_range,
                )
                node = self._refine_with_equity(node, mc_equity, spr)
                confidence = 0.70
            except ValueError:
                pass

        node = self._apply_position_adjustment(node, is_ip, hand_category)
        if num_opponents >= 2:
            node = self._apply_multiway_adjustment(node, num_opponents, hand_category)
        if opponent_stats is not None:
            node = self._apply_exploit_adjustment(node, opponent_stats, hand_category)

        return SolverResult(
            strategy=node,
            source="heuristic" if confidence < 0.65 else "monte_carlo",
            confidence=confidence,
            ev=node.weighted_ev,
            spot_key=spot_key,
        )

    def _lookup(
        self,
        board_bucket: str,
        position: str,
        hand_category: str,
        spr_bucket: str,
    ) -> StrategyNode | None:
        """Look up pre-computed postflop strategy."""
        bucket_data = self._data.get(board_bucket, {})
        category_data = bucket_data.get(hand_category, {})

        # Try exact SPR match, then "any"
        actions_data = category_data.get(spr_bucket) or category_data.get("any")
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

    def _heuristic_strategy(
        self,
        hand_category: str,
        board_bucket: str,
        pot: float,
        hero_stack: float,
        big_blind: float,
        spr: float,
    ) -> StrategyNode:
        """Build a heuristic strategy from the hand category and texture."""
        texture_type = _board_bucket_type(board_bucket)

        category_strategies = _HEURISTIC_STRATEGIES.get(hand_category, {})
        action_spec = category_strategies.get(texture_type)

        if action_spec is None:
            # Ultimate fallback
            action_spec = [("check", 0.50, 0.0), ("fold", 0.50, 0.0)]

        actions = []
        for action, freq, sizing_frac in action_spec:
            if action == "raise" and sizing_frac > 0:
                amount = BetSizingTree.compute_bet_amount(
                    pot, hero_stack, sizing_frac, big_blind * 2,
                )
            elif action == "call":
                amount = 0.0  # Will be set by caller based on actual bet
            else:
                amount = 0.0
            actions.append(ActionFrequency(action, freq, amount, 0.0))

        return StrategyNode(actions=actions)

    @staticmethod
    def _resolve_amounts(
        node: StrategyNode,
        pot: float,
        hero_stack: float,
        big_blind: float,
        board_bucket: str,
    ) -> StrategyNode:
        """Convert sizing fractions in pre-computed data to actual amounts."""
        resolved = []
        for af in node.actions:
            if af.action == "raise" and af.amount > 0 and af.amount <= 2.0:
                # Amount stored as pot fraction
                actual = BetSizingTree.compute_bet_amount(
                    pot, hero_stack, af.amount, big_blind * 2,
                )
                resolved.append(ActionFrequency(
                    af.action, af.frequency, actual, af.ev,
                ))
            else:
                resolved.append(af)
        return StrategyNode(actions=resolved)

    @staticmethod
    def _is_in_position(hero_position: str, villain_positions: list[str]) -> bool:
        """Determine if hero acts last (is in position) among active players.

        Postflop action order: SB(0), BB(1), UTG(2), MP(3), CO(4), BTN(5).
        Hero is IP if their order value is the highest among all active players.
        """
        order = PostflopSolver._POSTFLOP_ORDER
        hero_order = order.get(hero_position, 0)
        for vp in villain_positions:
            if order.get(vp, 0) > hero_order:
                return False
        return True

    @staticmethod
    def _apply_position_adjustment(
        node: StrategyNode, is_ip: bool, hand_category: str,
    ) -> StrategyNode:
        """Apply IP/OOP multipliers to strategy frequencies and sizing."""
        adjusted = []
        for af in node.actions:
            freq = af.frequency
            amount = af.amount

            if is_ip:
                if af.action == "raise":
                    freq *= 1.15
                    amount *= 0.85
                    if hand_category == "air":
                        freq *= 1.20
                elif af.action == "check":
                    freq *= 0.80
                # call and fold unchanged IP
            else:
                # OOP
                if af.action == "check":
                    freq *= 1.20
                elif af.action == "raise":
                    freq *= 0.85
                    amount *= 1.15
                elif af.action == "fold":
                    freq *= 1.10

            adjusted.append(ActionFrequency(af.action, freq, amount, af.ev))

        return StrategyNode(actions=adjusted).normalized()

    @staticmethod
    def _apply_multiway_adjustment(
        node: StrategyNode, num_opponents: int, hand_category: str,
    ) -> StrategyNode:
        """Apply multiway pot multipliers (num_opponents >= 2)."""
        raise_mult = 1.0 / (num_opponents ** 0.3)
        fold_mult = 1.0 + 0.15 * (num_opponents - 1)
        bluff_raise_mult = 1.0 / (num_opponents ** 0.5)

        adjusted = []
        for af in node.actions:
            freq = af.frequency
            amount = af.amount

            if af.action == "raise":
                if hand_category == "air":
                    freq *= bluff_raise_mult
                else:
                    freq *= raise_mult
                amount *= 0.85
            elif af.action == "fold":
                freq *= fold_mult
            # call unchanged

            adjusted.append(ActionFrequency(af.action, freq, amount, af.ev))

        return StrategyNode(actions=adjusted).normalized()

    @staticmethod
    def _effective_stat(
        observed: float,
        stat_name: str,
        sample_size: int,
    ) -> float:
        """Blend an observed stat toward GTO default based on sample confidence.

        Returns GTO default when sample_size < minimum threshold, linearly
        blends toward observed value, reaching full weight at 3× threshold.

        Args:
            observed: The raw observed stat value (e.g., VPIP 40%).
            stat_name: Key into _GTO_DEFAULTS / _MIN_SAMPLES.
            sample_size: Number of relevant observations.

        Returns:
            Blended stat value between GTO default and observed.
        """
        gto_default = _GTO_DEFAULTS[stat_name]
        threshold = _MIN_SAMPLES[stat_name]

        if sample_size < threshold:
            return gto_default

        full_at = threshold * _FULL_CONFIDENCE_MULT
        weight = min(1.0, (sample_size - threshold) / (full_at - threshold))
        return gto_default + weight * (observed - gto_default)

    @staticmethod
    def _apply_exploit_adjustment(
        node: StrategyNode,
        stats: OpponentStats,
        hand_category: str,
    ) -> StrategyNode:
        """Apply exploitative frequency shifts based on opponent tendencies.

        Adjusts bluff/value frequencies based on fold-to-cbet and aggression
        factor, with confidence gating via _effective_stat().
        """
        fold_cbet = PostflopSolver._effective_stat(
            stats.fold_to_cbet_pct, "fold_to_cbet", stats.cbet_faced,
        )
        agg_raw = min(stats.aggression_factor, 10.0)  # cap inf
        agg_sample = stats.aggression_actions + stats.passive_actions
        agg = PostflopSolver._effective_stat(
            agg_raw, "vpip", agg_sample,  # reuse vpip threshold (30) for AF
        )

        adjusted = []
        for af in node.actions:
            freq = af.frequency
            amount = af.amount

            if af.action == "raise":
                if hand_category == "air":
                    # Bluff more vs high folders, less vs low folders
                    if fold_cbet > 60:
                        freq *= 1.30
                    elif fold_cbet < 30:
                        freq *= 0.50
                else:
                    # Thin value bet more vs stations
                    if fold_cbet < 30:
                        freq *= 1.20
                # vs passive opponents, bluff more (they won't raise back)
                if agg < 1.0 and hand_category in ("air", "weak_draw"):
                    freq *= 1.15
                # Size down vs high folders (don't need big bets)
                if fold_cbet > 60 and amount > 0:
                    amount *= 0.85
            elif af.action == "check":
                # Trap more vs hyper-aggressive opponents
                if agg > 3.0 and hand_category in ("nuts", "strong_made"):
                    freq *= 1.25
            elif af.action == "fold":
                # Fold less vs passive opponents (they don't bluff)
                if agg < 1.0:
                    freq *= 0.85

            adjusted.append(ActionFrequency(af.action, freq, amount, af.ev))

        return StrategyNode(actions=adjusted).normalized()

    @staticmethod
    def _monte_carlo_equity(
        hero_cards: list[Card],
        community_cards: list[Card],
        opponent_range: Range,
        simulations: int = 1500,
    ) -> float:
        """Run Monte Carlo equity calculation.

        Uses parallel simulation for larger sim counts (>500).
        """
        result = EquityCalculator.parallel_hand_vs_range(
            hero_cards, opponent_range,
            board=community_cards,
            simulations=simulations,
        )
        return result.equity

    @staticmethod
    def _refine_with_equity(
        node: StrategyNode,
        equity: float,
        spr: float,
    ) -> StrategyNode:
        """Refine a heuristic strategy based on Monte Carlo equity."""
        adjusted = []
        for af in node.actions:
            if af.action in ("raise", "call"):
                # Boost aggressive actions if equity is high
                factor = 1.0 + (equity - 0.5) * 0.5
                new_freq = af.frequency * max(0.1, factor)
                adjusted.append(ActionFrequency(
                    af.action, new_freq, af.amount, equity * spr,
                ))
            elif af.action == "fold":
                # Reduce fold frequency if equity is decent
                factor = 1.0 - (equity - 0.3) * 0.3
                new_freq = af.frequency * max(0.05, factor)
                adjusted.append(ActionFrequency(
                    af.action, new_freq, af.amount, 0.0,
                ))
            else:
                adjusted.append(af)

        return StrategyNode(actions=adjusted).normalized()

    @staticmethod
    def _detect_street(community_cards: list[Card]) -> str:
        n = len(community_cards)
        if n <= 0:
            return "preflop"
        if n <= 3:
            return "flop"
        if n == 4:
            return "turn"
        return "river"

    @staticmethod
    def _action_seq(history: list[PriorAction]) -> str:
        """Summarize action sequence for spot key."""
        raises = sum(1 for a in history if a.action in (Action.RAISE, Action.ALL_IN))
        checks = sum(1 for a in history if a.action == Action.CHECK)
        if raises >= 2:
            return "raise_raise"
        if raises == 1:
            return "bet" if checks == 0 else "check_raise"
        if checks >= 1:
            return "checked"
        return "first_to_act"
