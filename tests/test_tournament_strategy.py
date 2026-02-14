"""Tests for tournament strategy adjustments."""

import pytest

from poker_bot.core.game_context import (
    GameContext,
    PayoutStructure,
    TournamentPhase,
)
from poker_bot.strategy.preflop_ranges import OPENING_RANGES, Range, get_opening_range
from poker_bot.strategy.tournament_strategy import (
    BubbleFactor,
    calculate_bubble_factor,
    calculate_icm,
    survival_premium,
    adjust_range_for_tournament,
)
from poker_bot.utils.constants import Position


class TestICM:
    def test_equal_stacks_equal_equity(self) -> None:
        stacks = [1000, 1000, 1000]
        payouts = [50, 30, 20]
        result = calculate_icm(stacks, payouts)
        # All players should have roughly equal equity (~33.3 each)
        for eq in result.equities:
            assert abs(eq - 33.33) < 1.0

    def test_chip_leader_has_most_equity(self) -> None:
        stacks = [5000, 3000, 2000]
        payouts = [50, 30, 20]
        result = calculate_icm(stacks, payouts)
        assert result.equities[0] > result.equities[1] > result.equities[2]

    def test_icm_less_than_chip_ev_for_leader(self) -> None:
        stacks = [8000, 1000, 1000]
        payouts = [50, 30, 20]
        result = calculate_icm(stacks, payouts)
        # Big stack's ICM equity should be less than chip-proportional equity
        chip_ev = result.chip_ev(0, 100)  # 80% of 100 = 80
        assert result.equities[0] < chip_ev

    def test_icm_more_than_chip_ev_for_short_stack(self) -> None:
        stacks = [8000, 1000, 1000]
        payouts = [50, 30, 20]
        result = calculate_icm(stacks, payouts)
        # Short stack's ICM equity should be more than chip-proportional
        chip_ev = result.chip_ev(1, 100)  # 10% of 100 = 10
        assert result.equities[1] > chip_ev

    def test_total_equity_equals_prize_pool(self) -> None:
        stacks = [5000, 3000, 2000]
        payouts = [50, 30, 20]
        result = calculate_icm(stacks, payouts)
        assert abs(result.total_equity - 100) < 0.5

    def test_heads_up_icm_equals_chip_ev(self) -> None:
        # With 2 players, ICM = chip EV
        stacks = [6000, 4000]
        payouts = [60, 40]
        result = calculate_icm(stacks, payouts)
        # Player 1: 60% of chips → 60% * 20 + 40 = 52
        # Actually HU ICM: p1 wins 60%, p2 wins 40%
        # p1 eq = 0.6*60 + 0.4*40 = 36+16 = 52
        # p2 eq = 0.4*60 + 0.6*40 = 24+24 = 48
        assert abs(result.equities[0] - 52) < 1.0
        assert abs(result.equities[1] - 48) < 1.0

    def test_zero_stack_zero_equity(self) -> None:
        stacks = [5000, 0, 5000]
        payouts = [50, 30, 20]
        result = calculate_icm(stacks, payouts)
        assert result.equities[1] == 0.0


class TestBubbleFactor:
    def test_equal_stacks_on_bubble(self) -> None:
        stacks = [1000, 1000, 1000, 1000]
        payouts = [50, 30, 20]  # 3 get paid, 4 players = bubble
        bf = calculate_bubble_factor(
            hero_stack=1000,
            villain_stack=1000,
            all_stacks=stacks,
            payouts=payouts,
        )
        # On the bubble with equal stacks, factor should be > 1
        assert bf.risk_factor > 1.0

    def test_big_stack_vs_short_stack(self) -> None:
        stacks = [5000, 1000, 2000, 2000]
        payouts = [50, 30, 20]
        bf = calculate_bubble_factor(
            hero_stack=5000,
            villain_stack=1000,
            all_stacks=stacks,
            payouts=payouts,
        )
        # Big stack risking chips against short stack — lower factor
        # than equal stacks
        assert bf.risk_factor > 0


class TestSurvivalPremium:
    def test_cash_game_no_premium(self) -> None:
        ctx = GameContext.cash_game(100)
        assert survival_premium(ctx) == 1.0

    def test_early_tournament_minimal_premium(self) -> None:
        ctx = GameContext.tournament(
            stack_bb=100,
            phase=TournamentPhase.EARLY,
            players_remaining=100,
        )
        assert survival_premium(ctx) >= 0.95

    def test_bubble_has_high_premium(self) -> None:
        ctx = GameContext.tournament(
            stack_bb=30,
            phase=TournamentPhase.BUBBLE,
            players_remaining=10,
        )
        premium = survival_premium(ctx)
        assert premium < 0.75

    def test_short_stack_on_bubble_tightest(self) -> None:
        bubble_avg = GameContext.tournament(
            stack_bb=30,
            phase=TournamentPhase.BUBBLE,
            players_remaining=10,
            average_stack_bb=30,
        )
        bubble_short = GameContext.tournament(
            stack_bb=10,
            phase=TournamentPhase.BUBBLE,
            players_remaining=10,
            average_stack_bb=30,
        )
        assert survival_premium(bubble_short) < survival_premium(bubble_avg)

    def test_big_stack_on_bubble_looser(self) -> None:
        bubble_avg = GameContext.tournament(
            stack_bb=30,
            phase=TournamentPhase.BUBBLE,
            players_remaining=10,
            average_stack_bb=30,
        )
        bubble_big = GameContext.tournament(
            stack_bb=80,
            phase=TournamentPhase.BUBBLE,
            players_remaining=10,
            average_stack_bb=30,
        )
        assert survival_premium(bubble_big) > survival_premium(bubble_avg)


class TestTournamentRangeAdjustment:
    def test_early_tournament_similar_to_cash(self) -> None:
        base = OPENING_RANGES[Position.BTN]
        ctx = GameContext.tournament(
            stack_bb=100,
            phase=TournamentPhase.EARLY,
            players_remaining=100,
        )
        adjusted = adjust_range_for_tournament(base, ctx)
        # Early tournament should be close to standard ranges
        assert adjusted.combo_count >= base.combo_count * 0.9

    def test_bubble_significantly_tighter(self) -> None:
        base = OPENING_RANGES[Position.BTN]
        ctx = GameContext.tournament(
            stack_bb=30,
            phase=TournamentPhase.BUBBLE,
            players_remaining=10,
        )
        adjusted = adjust_range_for_tournament(base, ctx)
        assert adjusted.combo_count < base.combo_count * 0.8

    def test_context_aware_opening_range_tournament(self) -> None:
        # Full integration: context-aware range via get_opening_range
        ctx = GameContext.tournament(
            stack_bb=30,
            phase=TournamentPhase.BUBBLE,
            players_remaining=10,
        )
        r = get_opening_range(Position.BTN, ctx)
        base = OPENING_RANGES[Position.BTN]
        assert r.combo_count < base.combo_count
