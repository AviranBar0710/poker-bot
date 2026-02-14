"""Tests for game context."""

from poker_bot.core.game_context import (
    BlindLevel,
    GameContext,
    GameType,
    PayoutStructure,
    TournamentPhase,
)


class TestGameContext:
    def test_cash_game_factory(self) -> None:
        ctx = GameContext.cash_game(100)
        assert ctx.game_type == GameType.CASH
        assert ctx.stack_depth_bb == 100
        assert ctx.is_cash
        assert not ctx.is_tournament

    def test_tournament_factory(self) -> None:
        ctx = GameContext.tournament(
            stack_bb=50,
            phase=TournamentPhase.MIDDLE,
            players_remaining=20,
        )
        assert ctx.is_tournament
        assert ctx.tournament_phase == TournamentPhase.MIDDLE

    def test_stack_categories(self) -> None:
        assert GameContext.cash_game(150).stack_category == "deep"
        assert GameContext.cash_game(100).stack_category == "deep"
        assert GameContext.cash_game(60).stack_category == "medium"
        assert GameContext.cash_game(30).stack_category == "short"
        assert GameContext.cash_game(15).stack_category == "very_short"
        assert GameContext.cash_game(8).stack_category == "critical"

    def test_m_ratio_with_blind_level(self) -> None:
        bl = BlindLevel(small_blind=100, big_blind=200, ante=25)
        ctx = GameContext.tournament(
            stack_bb=50,
            phase=TournamentPhase.MIDDLE,
            players_remaining=20,
            blind_level=bl,
        )
        # Stack = 50 * 200 = 10000
        # Pot preflop = 100 + 200 + 25*6 = 450
        # M = 10000 / 450 ≈ 22.2
        assert 22 < ctx.m_ratio < 23

    def test_m_ratio_without_blind_level(self) -> None:
        ctx = GameContext.cash_game(50)
        assert ctx.m_ratio == 50  # Falls back to stack_depth_bb

    def test_is_on_bubble(self) -> None:
        ctx = GameContext.tournament(
            stack_bb=30,
            phase=TournamentPhase.BUBBLE,
            players_remaining=10,
        )
        assert ctx.is_on_bubble

    def test_not_on_bubble(self) -> None:
        ctx = GameContext.tournament(
            stack_bb=30,
            phase=TournamentPhase.MIDDLE,
            players_remaining=20,
        )
        assert not ctx.is_on_bubble


class TestPayoutStructure:
    def test_min_cash_position(self) -> None:
        ps = PayoutStructure(
            total_prize_pool=1000,
            payouts={1: 500, 2: 300, 3: 200},
        )
        assert ps.min_cash_position == 3

    def test_payout_for(self) -> None:
        ps = PayoutStructure(
            total_prize_pool=1000,
            payouts={1: 500, 2: 300, 3: 200},
        )
        assert ps.payout_for(1) == 500
        assert ps.payout_for(4) == 0.0

    def test_remaining_payouts(self) -> None:
        ps = PayoutStructure(
            total_prize_pool=1000,
            payouts={1: 500, 2: 300, 3: 200},
        )
        remaining = ps.remaining_payouts(2)
        assert remaining == {1: 500, 2: 300}

    def test_is_near_payout_jump(self) -> None:
        ps = PayoutStructure(
            total_prize_pool=10000,
            payouts={1: 5000, 2: 3000, 3: 1200, 4: 800},
        )
        # At 3 players remaining: payout_for(3)=1200, payout_for(2)=3000
        # Jump ratio = 3000/1200 = 2.5 >= 1.5
        ctx = GameContext.tournament(
            stack_bb=30,
            phase=TournamentPhase.IN_THE_MONEY,
            players_remaining=3,
            payout_structure=ps,
        )
        assert ctx.is_near_payout_jump


class TestBlindLevel:
    def test_total_pot_preflop(self) -> None:
        bl = BlindLevel(small_blind=50, big_blind=100, ante=10)
        # 50 + 100 + 10*6 = 210
        assert bl.total_pot_preflop == 210

    def test_no_ante(self) -> None:
        bl = BlindLevel(small_blind=50, big_blind=100)
        assert bl.total_pot_preflop == 150
