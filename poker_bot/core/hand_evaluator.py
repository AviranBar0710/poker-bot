"""Texas Hold'em hand evaluation engine."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from poker_bot.utils.card import Card
from poker_bot.utils.constants import RANK_VALUES, HandRanking, Rank


@dataclass(frozen=True)
class HandResult:
    """Result of evaluating a poker hand."""

    ranking: HandRanking
    best_cards: tuple[Card, ...]
    kickers: tuple[Card, ...]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, HandResult):
            return NotImplemented
        if self.ranking != other.ranking:
            return self.ranking < other.ranking
        # Compare best cards by value (descending)
        for a, b in zip(
            sorted(self.best_cards, reverse=True),
            sorted(other.best_cards, reverse=True),
        ):
            if a.value != b.value:
                return a.value < b.value
        # Compare kickers
        for a, b in zip(
            sorted(self.kickers, reverse=True),
            sorted(other.kickers, reverse=True),
        ):
            if a.value != b.value:
                return a.value < b.value
        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandResult):
            return NotImplemented
        if self.ranking != other.ranking:
            return False
        self_vals = sorted([c.value for c in self.best_cards], reverse=True)
        other_vals = sorted([c.value for c in other.best_cards], reverse=True)
        if self_vals != other_vals:
            return False
        self_kick = sorted([c.value for c in self.kickers], reverse=True)
        other_kick = sorted([c.value for c in other.kickers], reverse=True)
        return self_kick == other_kick

    def __hash__(self) -> int:
        return hash((
            self.ranking,
            tuple(sorted(c.value for c in self.best_cards)),
            tuple(sorted(c.value for c in self.kickers)),
        ))

    def __le__(self, other: object) -> bool:
        if not isinstance(other, HandResult):
            return NotImplemented
        return self == other or self < other

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, HandResult):
            return NotImplemented
        return not self <= other

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, HandResult):
            return NotImplemented
        return not self < other


class HandEvaluator:
    """Evaluates poker hands and determines the best 5-card combination."""

    @staticmethod
    def evaluate(cards: list[Card]) -> HandResult:
        """Evaluate the best 5-card hand from a list of cards.

        Args:
            cards: 5 to 7 cards (hole cards + community cards).

        Returns:
            HandResult with the best hand ranking, cards, and kickers.

        Raises:
            ValueError: If fewer than 5 cards are provided.
        """
        if len(cards) < 5:
            raise ValueError(f"Need at least 5 cards, got {len(cards)}")

        best: HandResult | None = None
        for combo in combinations(cards, 5):
            result = HandEvaluator._evaluate_five(list(combo))
            if best is None or result > best:
                best = result
        assert best is not None
        return best

    @staticmethod
    def _evaluate_five(cards: list[Card]) -> HandResult:
        """Evaluate exactly 5 cards."""
        sorted_cards = sorted(cards, key=lambda c: c.value, reverse=True)
        is_flush = HandEvaluator._is_flush(sorted_cards)
        straight_high = HandEvaluator._straight_high(sorted_cards)
        is_straight = straight_high is not None
        rank_counts = Counter(c.rank for c in sorted_cards)
        counts = sorted(rank_counts.values(), reverse=True)

        if is_flush and is_straight:
            if straight_high == 14:
                return HandResult(
                    ranking=HandRanking.ROYAL_FLUSH,
                    best_cards=tuple(sorted_cards),
                    kickers=(),
                )
            return HandResult(
                ranking=HandRanking.STRAIGHT_FLUSH,
                best_cards=tuple(sorted_cards),
                kickers=(),
            )

        if counts == [4, 1]:
            return HandEvaluator._make_group_result(
                HandRanking.FOUR_OF_A_KIND, sorted_cards, rank_counts, 4
            )

        if counts == [3, 2]:
            return HandEvaluator._make_group_result(
                HandRanking.FULL_HOUSE, sorted_cards, rank_counts, 3
            )

        if is_flush:
            return HandResult(
                ranking=HandRanking.FLUSH,
                best_cards=tuple(sorted_cards),
                kickers=(),
            )

        if is_straight:
            return HandResult(
                ranking=HandRanking.STRAIGHT,
                best_cards=tuple(sorted_cards),
                kickers=(),
            )

        if counts == [3, 1, 1]:
            return HandEvaluator._make_group_result(
                HandRanking.THREE_OF_A_KIND, sorted_cards, rank_counts, 3
            )

        if counts == [2, 2, 1]:
            # Two pair: best cards are both pairs, kicker is the remaining card
            pairs = [r for r, c in rank_counts.items() if c == 2]
            pairs.sort(key=lambda r: RANK_VALUES[r], reverse=True)
            best = [c for c in sorted_cards if c.rank in pairs]
            kickers = [c for c in sorted_cards if c.rank not in pairs]
            return HandResult(
                ranking=HandRanking.TWO_PAIR,
                best_cards=tuple(best),
                kickers=tuple(kickers),
            )

        if counts == [2, 1, 1, 1]:
            return HandEvaluator._make_group_result(
                HandRanking.ONE_PAIR, sorted_cards, rank_counts, 2
            )

        return HandResult(
            ranking=HandRanking.HIGH_CARD,
            best_cards=tuple(sorted_cards[:1]),
            kickers=tuple(sorted_cards[1:]),
        )

    @staticmethod
    def _is_flush(cards: list[Card]) -> bool:
        """Check if all 5 cards share the same suit."""
        return len({c.suit for c in cards}) == 1

    @staticmethod
    def _straight_high(cards: list[Card]) -> int | None:
        """Return the high card value of a straight, or None.

        Handles the A-2-3-4-5 (wheel) straight as a special case.
        """
        values = sorted({c.value for c in cards}, reverse=True)
        if len(values) != 5:
            return None

        # Normal straight check
        if values[0] - values[4] == 4:
            return values[0]

        # Wheel: A-2-3-4-5
        if values == [14, 5, 4, 3, 2]:
            return 5

        return None

    @staticmethod
    def _make_group_result(
        ranking: HandRanking,
        sorted_cards: list[Card],
        rank_counts: Counter[Rank],
        group_size: int,
    ) -> HandResult:
        """Build a HandResult for group-based hands (pairs, trips, quads, full house)."""
        group_ranks = [r for r, c in rank_counts.items() if c == group_size]
        group_ranks.sort(key=lambda r: RANK_VALUES[r], reverse=True)
        best = [c for c in sorted_cards if c.rank in group_ranks]

        if ranking == HandRanking.FULL_HOUSE:
            # Kicker cards are the pair in a full house
            kicker_ranks = [r for r, c in rank_counts.items() if c != group_size]
            kickers = [c for c in sorted_cards if c.rank in kicker_ranks]
        else:
            kickers = [c for c in sorted_cards if c.rank not in group_ranks]

        return HandResult(
            ranking=ranking,
            best_cards=tuple(best),
            kickers=tuple(kickers),
        )
