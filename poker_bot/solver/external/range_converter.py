"""Convert Range objects to external solver input formats.

Pure data transformation — no strategy logic, no heuristics.
"""

from __future__ import annotations

from poker_bot.strategy.preflop_ranges import Range
from poker_bot.utils.card import Card


class RangeConverter:
    """Convert between our Range objects and solver-specific formats."""

    @staticmethod
    def to_texas_solver(range_obj: Range) -> str:
        """Convert Range to TexasSolver comma-separated notation.

        Output format: "AA,KK,QQ,AKs,AKo,T9s,..."

        TexasSolver handles board card removal internally, so we
        emit every HandNotation in the range without dead-card filtering.

        Args:
            range_obj: Our internal Range object.

        Returns:
            Comma-separated hand notation string for TexasSolver.
        """
        if not range_obj.hands:
            return ""

        # Sort for deterministic output (pairs first, then suited, then offsuit)
        notations = sorted(str(h) for h in range_obj.hands)
        return ",".join(notations)

    @staticmethod
    def to_piosolver(range_obj: Range, dead_cards: list[Card] | None = None) -> list[float]:
        """Convert Range to PioSolver 1326-float array.

        Each float is 0.0 (not in range) or 1.0 (in range).
        Index order follows PioSolver's canonical combo ordering.

        Note: This is a stub for future PioSolver integration.
        The canonical 1326 ordering needs to match PioSolver's
        `show_hand_order` output exactly.

        Args:
            range_obj: Our internal Range object.
            dead_cards: Cards to exclude (hero cards + board).

        Returns:
            List of 1326 floats.
        """
        dead_set = set(dead_cards) if dead_cards else set()

        # Build the canonical 52-card ordering: 2c,2d,2h,2s,...,Ac,Ad,Ah,As
        from poker_bot.utils.constants import Rank, Suit

        card_order = []
        rank_order = [
            Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
            Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
            Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE,
        ]
        suit_order = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
        for rank in rank_order:
            for suit in suit_order:
                card_order.append(Card(rank, suit))

        card_index = {c: i for i, c in enumerate(card_order)}

        # Build combo set from range
        combo_set: set[tuple[int, int]] = set()
        for combo in range_obj.to_combos():
            c1 = combo.card1
            c2 = combo.card2
            if c1 in dead_set or c2 in dead_set:
                continue
            i1 = card_index.get(c1)
            i2 = card_index.get(c2)
            if i1 is not None and i2 is not None:
                lo, hi = min(i1, i2), max(i1, i2)
                combo_set.add((lo, hi))

        # Build 1326-float array using triangular indexing
        result = [0.0] * 1326
        idx = 0
        for i in range(52):
            for j in range(i + 1, 52):
                if (i, j) in combo_set:
                    result[idx] = 1.0
                idx += 1

        return result
