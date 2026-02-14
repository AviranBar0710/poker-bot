"""Tests for the equity calculator."""

import pytest

from poker_bot.core.equity_calculator import EquityCalculator
from poker_bot.strategy.preflop_ranges import Range
from poker_bot.utils.card import Card


def _cards(s: str) -> list[Card]:
    return [Card.from_str(c) for c in s.split()]


class TestHandVsHand:
    def test_aa_vs_kk_preflop(self) -> None:
        result = EquityCalculator.hand_vs_hand(
            _cards("Ah As"), _cards("Kh Ks"), simulations=5_000
        )
        # AA vs KK is ~82% equity
        assert 0.75 < result.equity < 0.90

    def test_aa_vs_72o_dominant(self) -> None:
        result = EquityCalculator.hand_vs_hand(
            _cards("Ah As"), _cards("7h 2c"), simulations=5_000
        )
        assert result.equity > 0.80

    def test_coinflip_pair_vs_overcards(self) -> None:
        result = EquityCalculator.hand_vs_hand(
            _cards("Jh Js"), _cards("Ah Ks"), simulations=5_000
        )
        # JJ vs AKo is ~55% for JJ
        assert 0.45 < result.equity < 0.65

    def test_with_flop(self) -> None:
        # AA on a K-high board should still be strong
        result = EquityCalculator.hand_vs_hand(
            _cards("Ah As"),
            _cards("Kh Qs"),
            board=_cards("Kc 7d 2s"),
            simulations=5_000,
        )
        assert result.equity > 0.50

    def test_made_hand_on_river(self) -> None:
        # Full board dealt — deterministic result
        result = EquityCalculator.hand_vs_hand(
            _cards("Ah Kh"),
            _cards("Qs Qd"),
            board=_cards("Ac 7d 2s 8c 3h"),
            simulations=100,
        )
        # AK made top pair, QQ is second pair — AK wins
        assert result.equity == 1.0

    def test_result_counts_sum(self) -> None:
        result = EquityCalculator.hand_vs_hand(
            _cards("Ah Kh"), _cards("Qh Jh"), simulations=1_000
        )
        assert result.win_count + result.tie_count + result.loss_count == 1_000


class TestHandVsRange:
    def test_aa_vs_wide_range(self) -> None:
        wide = Range().add("KK,QQ,JJ,TT,AKs,AKo,AQs")
        result = EquityCalculator.hand_vs_range(
            _cards("Ah As"), wide, simulations=3_000
        )
        # AA vs a strong range should still be ahead
        assert result.equity > 0.60

    def test_with_board(self) -> None:
        opponent = Range().add("AA,KK,QQ,AKs")
        result = EquityCalculator.hand_vs_range(
            _cards("Jh Js"),
            opponent,
            board=_cards("Jd 7c 2s"),  # We flopped a set!
            simulations=3_000,
        )
        assert result.equity > 0.70

    def test_empty_range_after_filter_raises(self) -> None:
        # Range of only AhAs — impossible if we hold those exact cards
        # Use a single suited combo and block both cards
        r = Range()
        from poker_bot.strategy.preflop_ranges import HandCombo, HandNotation, HandType
        from poker_bot.utils.constants import Rank

        # Create a range with only one combo: Ah-Kh
        h = HandNotation(Rank.ACE, Rank.KING, HandType.SUITED)
        r.hands.add(h)
        # We hold Ah + Kh, so all 4 suited AK combos where suit matches are blocked
        # But AKs has 4 combos (one per suit), and we only block the hearts combo
        # Instead, just test that the calculator works correctly by providing
        # a hand that blocks ALL combos in a tiny range
        r2 = Range()
        r2.hands.add(HandNotation(Rank.ACE, Rank.ACE, HandType.PAIR))
        # Hold all 4 aces to block every AA combo
        with pytest.raises(ValueError, match="No valid combos"):
            EquityCalculator.hand_vs_range(
                _cards("Ah As"), r2, board=_cards("Ac Ad 2s"), simulations=100
            )


class TestRangeVsRange:
    def test_tight_vs_wide(self) -> None:
        tight = Range().add("AA,KK,QQ")
        wide = Range().add("TT,99,88,77,AJs,ATs,KQs,QJs,JTs")
        result = EquityCalculator.range_vs_range(tight, wide, simulations=3_000)
        # Premium pairs should have good equity vs a wide range
        assert result.equity > 0.55

    def test_with_board(self) -> None:
        r1 = Range().add("AA,KK")
        r2 = Range().add("QQ,JJ,TT")
        result = EquityCalculator.range_vs_range(
            r1, r2, board=_cards("2c 5d 8h"), simulations=3_000
        )
        assert result.equity > 0.55
