"""Tests for the hand evaluator."""

from poker_bot.core.hand_evaluator import HandEvaluator, HandResult
from poker_bot.utils.card import Card
from poker_bot.utils.constants import HandRanking


def _cards(s: str) -> list[Card]:
    """Helper: parse space-separated card strings like 'Ah Kh Qh Jh Th'."""
    return [Card.from_str(c) for c in s.split()]


class TestHandRankings:
    def test_royal_flush(self) -> None:
        result = HandEvaluator.evaluate(_cards("Ah Kh Qh Jh Th"))
        assert result.ranking == HandRanking.ROYAL_FLUSH

    def test_straight_flush(self) -> None:
        result = HandEvaluator.evaluate(_cards("9s 8s 7s 6s 5s"))
        assert result.ranking == HandRanking.STRAIGHT_FLUSH

    def test_straight_flush_wheel(self) -> None:
        result = HandEvaluator.evaluate(_cards("5d 4d 3d 2d Ad"))
        assert result.ranking == HandRanking.STRAIGHT_FLUSH

    def test_four_of_a_kind(self) -> None:
        result = HandEvaluator.evaluate(_cards("Ks Kh Kd Kc 3s"))
        assert result.ranking == HandRanking.FOUR_OF_A_KIND

    def test_full_house(self) -> None:
        result = HandEvaluator.evaluate(_cards("Jh Jd Jc 8s 8h"))
        assert result.ranking == HandRanking.FULL_HOUSE

    def test_flush(self) -> None:
        result = HandEvaluator.evaluate(_cards("Ah Th 7h 4h 2h"))
        assert result.ranking == HandRanking.FLUSH

    def test_straight(self) -> None:
        result = HandEvaluator.evaluate(_cards("9h 8s 7d 6c 5h"))
        assert result.ranking == HandRanking.STRAIGHT

    def test_straight_wheel(self) -> None:
        result = HandEvaluator.evaluate(_cards("5h 4s 3d 2c Ah"))
        assert result.ranking == HandRanking.STRAIGHT

    def test_three_of_a_kind(self) -> None:
        result = HandEvaluator.evaluate(_cards("Qs Qh Qd 7c 3s"))
        assert result.ranking == HandRanking.THREE_OF_A_KIND

    def test_two_pair(self) -> None:
        result = HandEvaluator.evaluate(_cards("As Ah 8d 8c 4s"))
        assert result.ranking == HandRanking.TWO_PAIR

    def test_one_pair(self) -> None:
        result = HandEvaluator.evaluate(_cards("Ts Th 9d 5c 2s"))
        assert result.ranking == HandRanking.ONE_PAIR

    def test_high_card(self) -> None:
        result = HandEvaluator.evaluate(_cards("Ah Ks 9d 5c 2h"))
        assert result.ranking == HandRanking.HIGH_CARD


class TestHandComparison:
    def test_flush_beats_straight(self) -> None:
        flush = HandEvaluator.evaluate(_cards("Ah Th 7h 4h 2h"))
        straight = HandEvaluator.evaluate(_cards("9h 8s 7d 6c 5h"))
        assert flush > straight

    def test_higher_pair_wins(self) -> None:
        aces = HandEvaluator.evaluate(_cards("As Ah Kd 7c 3s"))
        kings = HandEvaluator.evaluate(_cards("Ks Kh Ad 7c 3s"))
        assert aces > kings

    def test_kicker_decides_pair(self) -> None:
        high_kicker = HandEvaluator.evaluate(_cards("As Ah Kd 7c 3s"))
        low_kicker = HandEvaluator.evaluate(_cards("As Ah Qd 7c 3s"))
        assert high_kicker > low_kicker

    def test_equal_hands(self) -> None:
        hand1 = HandEvaluator.evaluate(_cards("As Kh Qd Jc 9s"))
        hand2 = HandEvaluator.evaluate(_cards("Ah Ks Qc Jd 9h"))
        assert hand1 == hand2


class TestSevenCardEvaluation:
    def test_best_five_from_seven(self) -> None:
        # Hole cards: Ah Kh, Board: Qh Jh Th 3c 2d → royal flush
        result = HandEvaluator.evaluate(_cards("Ah Kh Qh Jh Th 3c 2d"))
        assert result.ranking == HandRanking.ROYAL_FLUSH

    def test_picks_flush_over_pair(self) -> None:
        # Hole cards: Ah 9h, Board: 7h 4h 2h Ks Kd
        result = HandEvaluator.evaluate(_cards("Ah 9h 7h 4h 2h Ks Kd"))
        assert result.ranking == HandRanking.FLUSH

    def test_picks_full_house_from_board(self) -> None:
        # Hole cards: Js Jh, Board: Jd 8s 8h 3c 2d
        result = HandEvaluator.evaluate(_cards("Js Jh Jd 8s 8h 3c 2d"))
        assert result.ranking == HandRanking.FULL_HOUSE

    def test_too_few_cards_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Need at least 5 cards"):
            HandEvaluator.evaluate(_cards("Ah Kh Qh"))
