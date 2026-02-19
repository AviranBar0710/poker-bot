"""Action-history-based opponent range narrowing.

Replaces simple raise-count logic with street-by-street range narrowing
that considers position, action type, bet sizing, and board texture.
"""

from __future__ import annotations

from poker_bot.strategy.decision_maker import PriorAction
from poker_bot.strategy.preflop_ranges import (
    CALL_VS_RAISE_RANGES,
    FOUR_BET_RANGES,
    OPENING_RANGES,
    THREE_BET_RANGES,
    Range,
)
from poker_bot.utils.constants import Action, Position, Street


class RangeEstimator:
    """Estimates opponent ranges based on action history.

    Narrows ranges street by street based on position-aware opening
    ranges, aggression level, and bet sizing tells.
    """

    @staticmethod
    def estimate_preflop_range(
        villain_position: Position | None,
        action_history: list[PriorAction],
    ) -> Range:
        """Estimate opponent's preflop range from their actions.

        Args:
            villain_position: Villain's table position (None = unknown).
            action_history: All preflop actions.

        Returns:
            Estimated range for the villain.
        """
        # Determine villain's aggression level
        villain_raises = 0
        villain_called = False
        for a in action_history:
            if villain_position and a.position != villain_position:
                continue
            if a.action in (Action.RAISE, Action.ALL_IN):
                villain_raises += 1
            elif a.action == Action.CALL:
                villain_called = True

        pos = villain_position or Position.CO  # Default to CO if unknown

        if villain_raises >= 3:
            # 4-bet+: extremely narrow
            return FOUR_BET_RANGES.get(pos, Range().add("AA,KK"))

        if villain_raises >= 2:
            # 3-bet: strong range
            return THREE_BET_RANGES.get(pos, Range().add("AA,KK,QQ,AKs"))

        if villain_raises >= 1:
            # Open raise: position-based range
            return OPENING_RANGES.get(pos, Range().add(
                "AA,KK,QQ,JJ,TT,99,88,77,AKs,AQs,AJs,ATs,AKo,AQo,KQs"
            ))

        if villain_called:
            # Cold call or limp: call ranges or wide
            return CALL_VS_RAISE_RANGES.get(pos, Range().add(
                "TT,99,88,77,66,AQs,AJs,ATs,KQs,KJs,QJs,JTs,T9s,98s,87s"
            ))

        # Checked or limped: wide range
        return Range().add(
            "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,"
            "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
            "AKo,AQo,AJo,ATo,A9o,A8o,A7o,"
            "KQs,KJs,KTs,K9s,K8s,K7s,"
            "KQo,KJo,KTo,"
            "QJs,QTs,Q9s,Q8s,"
            "QJo,QTo,"
            "JTs,J9s,J8s,JTo,"
            "T9s,T8s,98s,97s,87s,86s,76s,75s,65s,64s,54s"
        )

    @staticmethod
    def narrow_for_postflop_action(
        base_range: Range,
        action: Action,
        bet_size_fraction: float,
        street: Street,
    ) -> Range:
        """Narrow a range based on a postflop action.

        Aggressive actions narrow the range (strong hands + some bluffs).
        Passive actions keep a wider range.

        Args:
            base_range: Opponent's current estimated range.
            action: The action taken.
            bet_size_fraction: Bet size as fraction of pot.
            street: Which street the action occurred on.

        Returns:
            Narrowed range.
        """
        if not base_range.hands:
            return base_range

        from poker_bot.strategy.tournament_strategy import _hand_strength_key

        sorted_hands = sorted(
            base_range.hands, key=_hand_strength_key, reverse=True,
        )

        total = len(sorted_hands)
        if total == 0:
            return base_range

        if action in (Action.RAISE, Action.ALL_IN):
            # Raising narrows significantly
            # Larger sizing = more polarized (stronger or bluff)
            if bet_size_fraction >= 0.75:
                keep_pct = 0.35
            elif bet_size_fraction >= 0.5:
                keep_pct = 0.45
            else:
                keep_pct = 0.55
        elif action == Action.CALL:
            # Calling keeps medium-strength hands
            keep_pct = 0.70
        elif action == Action.CHECK:
            # Checking doesn't narrow much but removes some strength
            keep_pct = 0.85
        else:
            return base_range  # Fold = no range left (shouldn't happen)

        # Later streets narrow more
        street_factor = {
            Street.FLOP: 1.0,
            Street.TURN: 0.85,
            Street.RIVER: 0.70,
        }.get(street, 1.0)

        keep_count = max(1, int(total * keep_pct * street_factor))

        if action in (Action.RAISE, Action.ALL_IN):
            # Polarized: top portion + some bottom (bluffs)
            value_count = max(1, int(keep_count * 0.7))
            bluff_count = keep_count - value_count
            kept = set(sorted_hands[:value_count])
            if bluff_count > 0 and total > value_count:
                kept.update(sorted_hands[-bluff_count:])
        else:
            # Linear: keep the top hands
            kept = set(sorted_hands[:keep_count])

        return Range(hands=kept)

    @staticmethod
    def categorize_hand(
        hand_strength: float,
        has_draw: bool,
        draw_strength: float = 0.0,
    ) -> str:
        """Categorize a hand into strategic buckets.

        Args:
            hand_strength: Made-hand strength [0, 1].
            has_draw: Whether the hand has a significant draw.
            draw_strength: How strong the draw is [0, 1].

        Returns:
            Category string: nuts, strong_made, medium_made,
            weak_made, strong_draw, medium_draw, weak_draw, air.
        """
        if hand_strength >= 0.95:
            return "nuts"
        if hand_strength >= 0.85:
            return "strong_made"
        if hand_strength >= 0.65:
            return "medium_made"
        if hand_strength >= 0.40:
            if has_draw and draw_strength >= 0.5:
                return "strong_draw"
            return "weak_made"
        if has_draw:
            if draw_strength >= 0.5:
                return "strong_draw"
            if draw_strength >= 0.25:
                return "medium_draw"
            return "weak_draw"
        if hand_strength >= 0.15:
            return "weak_draw"
        return "air"
