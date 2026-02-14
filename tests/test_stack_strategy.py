"""Tests for stack-depth strategy adjustments."""

from poker_bot.core.game_context import GameContext
from poker_bot.strategy.preflop_ranges import get_opening_range, OPENING_RANGES
from poker_bot.strategy.stack_strategy import (
    adjust_range_for_stack,
    get_push_fold_range,
)
from poker_bot.utils.constants import Position


class TestPushFoldRanges:
    def test_returns_none_for_deep_stack(self) -> None:
        assert get_push_fold_range(50, Position.BTN) is None

    def test_returns_range_for_10bb(self) -> None:
        r = get_push_fold_range(10, Position.BTN)
        assert r is not None
        assert r.combo_count > 0

    def test_returns_range_for_5bb(self) -> None:
        r = get_push_fold_range(5, Position.UTG)
        assert r is not None
        assert r.combo_count > 0

    def test_btn_wider_than_utg_at_10bb(self) -> None:
        utg = get_push_fold_range(10, Position.UTG)
        btn = get_push_fold_range(10, Position.BTN)
        assert utg is not None and btn is not None
        assert utg.combo_count < btn.combo_count

    def test_5bb_tighter_than_10bb(self) -> None:
        r10 = get_push_fold_range(10, Position.BTN)
        r5 = get_push_fold_range(5, Position.BTN)
        assert r10 is not None and r5 is not None
        assert r5.combo_count < r10.combo_count


class TestStackAdjustments:
    def test_deep_stack_adds_hands(self) -> None:
        base = OPENING_RANGES[Position.UTG]
        ctx = GameContext.cash_game(150)
        adjusted = adjust_range_for_stack(base, ctx)
        assert adjusted.combo_count >= base.combo_count

    def test_medium_stack_removes_hands(self) -> None:
        base = OPENING_RANGES[Position.BTN]
        ctx = GameContext.cash_game(60)
        adjusted = adjust_range_for_stack(base, ctx)
        assert adjusted.combo_count <= base.combo_count

    def test_short_stack_tighter_than_medium(self) -> None:
        base = OPENING_RANGES[Position.BTN]
        medium_ctx = GameContext.cash_game(60)
        short_ctx = GameContext.cash_game(25)
        medium = adjust_range_for_stack(base, medium_ctx)
        short = adjust_range_for_stack(base, short_ctx)
        assert short.combo_count <= medium.combo_count

    def test_100bb_unchanged(self) -> None:
        # 100bb is "deep" category, but the base ranges ARE 100bb ranges
        # Deep stack adds speculative hands
        base = OPENING_RANGES[Position.UTG]
        ctx = GameContext.cash_game(100)
        adjusted = adjust_range_for_stack(base, ctx)
        # Deep adds hands, so adjusted >= base
        assert adjusted.combo_count >= base.combo_count


class TestContextAwareRanges:
    def test_context_none_returns_base(self) -> None:
        base = get_opening_range(Position.UTG)
        assert base.combo_count == OPENING_RANGES[Position.UTG].combo_count

    def test_deep_cash_game(self) -> None:
        ctx = GameContext.cash_game(200)
        r = get_opening_range(Position.UTG, ctx)
        base = OPENING_RANGES[Position.UTG]
        assert r.combo_count >= base.combo_count

    def test_short_stack_tighter(self) -> None:
        ctx = GameContext.cash_game(25)
        r = get_opening_range(Position.BTN, ctx)
        base = OPENING_RANGES[Position.BTN]
        assert r.combo_count < base.combo_count

    def test_critical_stack_uses_push_fold(self) -> None:
        ctx = GameContext.cash_game(8)
        r = get_opening_range(Position.BTN, ctx)
        pf = get_push_fold_range(8, Position.BTN)
        assert pf is not None
        # Should use push/fold range
        assert r.combo_count == pf.combo_count
