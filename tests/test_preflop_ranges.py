"""Tests for the pre-flop range system."""

import pytest

from poker_bot.strategy.preflop_ranges import (
    OPENING_RANGES,
    THREE_BET_RANGES,
    FOUR_BET_RANGES,
    CALL_VS_RAISE_RANGES,
    HandNotation,
    HandType,
    Range,
    expand_notation,
    get_opening_range,
)
from poker_bot.utils.card import Card
from poker_bot.utils.constants import Position, Rank


class TestHandNotation:
    def test_parse_pair(self) -> None:
        h = HandNotation.from_str("AA")
        assert h.rank1 == Rank.ACE
        assert h.rank2 == Rank.ACE
        assert h.hand_type == HandType.PAIR

    def test_parse_suited(self) -> None:
        h = HandNotation.from_str("AKs")
        assert h.rank1 == Rank.ACE
        assert h.rank2 == Rank.KING
        assert h.hand_type == HandType.SUITED

    def test_parse_offsuit(self) -> None:
        h = HandNotation.from_str("AKo")
        assert h.rank1 == Rank.ACE
        assert h.rank2 == Rank.KING
        assert h.hand_type == HandType.OFFSUIT

    def test_parse_normalizes_rank_order(self) -> None:
        h = HandNotation.from_str("KAs")
        assert h.rank1 == Rank.ACE
        assert h.rank2 == Rank.KING

    def test_pair_has_6_combos(self) -> None:
        h = HandNotation.from_str("AA")
        assert h.combo_count == 6
        assert len(h.to_combos()) == 6

    def test_suited_has_4_combos(self) -> None:
        h = HandNotation.from_str("AKs")
        assert h.combo_count == 4
        assert len(h.to_combos()) == 4

    def test_offsuit_has_12_combos(self) -> None:
        h = HandNotation.from_str("AKo")
        assert h.combo_count == 12
        assert len(h.to_combos()) == 12

    def test_invalid_notation(self) -> None:
        with pytest.raises(ValueError):
            HandNotation.from_str("AKx")


class TestExpandNotation:
    def test_plus_pairs(self) -> None:
        hands = expand_notation("JJ+")
        ranks = {h.rank1 for h in hands}
        assert ranks == {Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE}

    def test_plus_suited(self) -> None:
        hands = expand_notation("ATs+")
        assert len(hands) == 4  # ATs, AJs, AQs, AKs
        for h in hands:
            assert h.rank1 == Rank.ACE
            assert h.hand_type == HandType.SUITED

    def test_dash_pairs(self) -> None:
        hands = expand_notation("JJ-88")
        assert len(hands) == 4  # JJ, TT, 99, 88

    def test_dash_suited(self) -> None:
        hands = expand_notation("A5s-A2s")
        assert len(hands) == 4  # A5s, A4s, A3s, A2s

    def test_single_with_both_suits(self) -> None:
        hands = expand_notation("AK")
        assert len(hands) == 2  # AKs and AKo

    def test_invalid_dash_different_high(self) -> None:
        with pytest.raises(ValueError, match="share the high card"):
            expand_notation("A5s-K2s")


class TestRange:
    def test_add_and_count(self) -> None:
        r = Range().add("AA,KK,QQ")
        assert len(r) == 3
        assert r.combo_count == 18  # 3 pairs * 6 combos

    def test_remove(self) -> None:
        r = Range().add("AA,KK,QQ").remove("KK")
        assert len(r) == 2
        assert r.combo_count == 12

    def test_percentage(self) -> None:
        r = Range().add("AA")
        # AA = 6 combos / 1326 total
        assert abs(r.percentage - (6 / 1326 * 100)) < 0.01

    def test_contains_card(self) -> None:
        r = Range().add("AA")
        ah = Card.from_str("Ah")
        ac = Card.from_str("Ac")
        kh = Card.from_str("Kh")
        assert r.contains(ah, ac)
        assert not r.contains(ah, kh)

    def test_chaining(self) -> None:
        r = Range().add("AA").add("KK").remove("AA")
        assert len(r) == 1


class TestGTORanges:
    def test_utg_tighter_than_btn(self) -> None:
        utg = OPENING_RANGES[Position.UTG]
        btn = OPENING_RANGES[Position.BTN]
        assert utg.combo_count < btn.combo_count

    def test_position_order_gets_wider(self) -> None:
        positions = [Position.UTG, Position.MP, Position.CO, Position.BTN]
        sizes = [OPENING_RANGES[p].combo_count for p in positions]
        for i in range(len(sizes) - 1):
            assert sizes[i] <= sizes[i + 1], (
                f"{positions[i]} should be tighter than {positions[i + 1]}"
            )

    def test_3bet_subset_conceptually_sound(self) -> None:
        # 3-bet range should be smaller than open range for same position
        for pos in [Position.UTG, Position.MP, Position.CO, Position.BTN]:
            assert THREE_BET_RANGES[pos].combo_count < OPENING_RANGES[pos].combo_count

    def test_4bet_tighter_than_3bet(self) -> None:
        for pos in [Position.UTG, Position.MP, Position.CO, Position.BTN]:
            assert FOUR_BET_RANGES[pos].combo_count <= THREE_BET_RANGES[pos].combo_count

    def test_bb_has_no_open_range(self) -> None:
        r = get_opening_range(Position.BB)
        assert len(r) == 0

    def test_all_ranges_have_reasonable_size(self) -> None:
        for pos, rng in OPENING_RANGES.items():
            pct = rng.percentage
            assert 10 < pct < 60, f"{pos} open range is {pct:.1f}%"
