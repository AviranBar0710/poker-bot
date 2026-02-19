"""Integration tests for solver engine and DecisionMaker with solver injection."""

import time

from poker_bot.core.game_context import GameContext, TournamentPhase
from poker_bot.core.game_state import GameState, PlayerState
from poker_bot.solver.data_structures import ActionFrequency, StrategyNode
from poker_bot.solver.engine import SolverEngine
from poker_bot.solver.icm_adapter import adjust_for_icm
from poker_bot.strategy.decision_maker import (
    ActionType,
    DecisionMaker,
    PriorAction,
)
from poker_bot.utils.card import Card
from poker_bot.utils.constants import Action, Position, Street


def _card(s: str) -> Card:
    return Card.from_str(s)


def _cards(s: str) -> list[Card]:
    return [Card.from_str(c) for c in s.split()]


def _make_game_state(
    hero_cards: list[Card],
    position: Position = Position.BTN,
    stack: float = 100.0,
    pot: float = 3.0,
    current_bet: float = 0.0,
    street: Street = Street.PREFLOP,
    community_cards: list[Card] | None = None,
) -> GameState:
    hero = PlayerState(
        name="Hero", chips=stack, position=position,
        hole_cards=hero_cards, is_active=True,
    )
    villain = PlayerState(
        name="Villain", chips=stack, position=Position.UTG,
        hole_cards=[], is_active=True,
    )
    return GameState(
        players=[hero, villain],
        small_blind=0.5,
        big_blind=1.0,
        pot=pot,
        current_bet=current_bet,
        current_street=street,
        community_cards=community_cards or [],
    )


class TestSolverEngine:
    def setup_method(self):
        self.engine = SolverEngine()

    def test_preflop_solve(self):
        gs = _make_game_state(_cards("Ah Kh"))
        ctx = GameContext.cash_game(100.0)
        result = self.engine.solve(gs, ctx, 0)
        assert result.strategy.recommended_action is not None
        assert result.confidence > 0
        assert result.source in ("preflop_lookup", "heuristic")

    def test_postflop_solve(self):
        gs = _make_game_state(
            _cards("Ah Kh"),
            street=Street.FLOP,
            community_cards=_cards("Qh Jd 3c"),
            pot=10.0,
        )
        ctx = GameContext.cash_game(100.0)
        result = self.engine.solve(gs, ctx, 0)
        assert result.strategy.recommended_action is not None
        assert result.spot_key is not None
        assert result.spot_key.street == "flop"

    def test_tournament_icm_adjustment(self):
        gs = _make_game_state(_cards("8h 7h"))
        ctx_cash = GameContext.cash_game(100.0)
        ctx_bubble = GameContext.tournament(
            stack_bb=30.0,
            phase=TournamentPhase.BUBBLE,
            players_remaining=20,
        )

        result_cash = self.engine.solve(gs, ctx_cash, 0)
        result_bubble = self.engine.solve(gs, ctx_bubble, 0)

        # Bubble should have higher fold frequency
        def fold_freq(result):
            for a in result.strategy.actions:
                if a.action == "fold":
                    return a.frequency
            return 0.0

        assert fold_freq(result_bubble) >= fold_freq(result_cash)

    def test_performance_under_2s(self):
        """Solver should complete in under 2 seconds for any spot."""
        gs = _make_game_state(
            _cards("Jh Th"),
            street=Street.FLOP,
            community_cards=_cards("9h 8d 2c"),
            pot=10.0,
        )
        ctx = GameContext.cash_game(100.0)

        start = time.time()
        result = self.engine.solve(gs, ctx, 0)
        elapsed = time.time() - start

        assert elapsed < 2.0
        assert result.strategy.recommended_action is not None

    def test_no_cards_returns_fold(self):
        hero = PlayerState(
            name="Hero", chips=100.0, position=Position.BTN,
            hole_cards=[], is_active=True,
        )
        gs = GameState(
            players=[hero],
            small_blind=0.5, big_blind=1.0, pot=3.0,
        )
        ctx = GameContext.cash_game(100.0)
        result = self.engine.solve(gs, ctx, 0)
        assert result.strategy.recommended_action.action == "fold"
        assert result.confidence == 0.0


class TestICMAdapter:
    def test_no_adjustment_for_cash(self):
        node = StrategyNode(actions=[
            ActionFrequency("raise", 0.7, 6.0),
            ActionFrequency("fold", 0.3, 0.0),
        ])
        adjusted = adjust_for_icm(node, 1.0)
        assert adjusted.actions[0].frequency == 0.7

    def test_increases_fold_on_bubble(self):
        node = StrategyNode(actions=[
            ActionFrequency("raise", 0.7, 6.0),
            ActionFrequency("fold", 0.3, 0.0),
        ])
        adjusted = adjust_for_icm(node, 0.6)
        fold_freq = next(
            a.frequency for a in adjusted.actions if a.action == "fold"
        )
        assert fold_freq > 0.3

    def test_frequencies_normalized(self):
        node = StrategyNode(actions=[
            ActionFrequency("raise", 0.5, 6.0),
            ActionFrequency("call", 0.3, 2.0),
            ActionFrequency("fold", 0.2, 0.0),
        ])
        adjusted = adjust_for_icm(node, 0.5)
        total = sum(a.frequency for a in adjusted.actions)
        assert abs(total - 1.0) < 0.01


class TestDecisionMakerWithSolver:
    def test_solver_none_works_as_before(self):
        """DecisionMaker(solver=None) should work exactly as before."""
        maker = DecisionMaker()
        gs = _make_game_state(_cards("Ah As"))
        ctx = GameContext.cash_game(100.0)
        decision = maker.make_decision(gs, ctx, 0)
        assert decision.action in (ActionType.RAISE, ActionType.ALL_IN)

    def test_solver_injected(self):
        """DecisionMaker with solver should use solver output."""
        engine = SolverEngine()
        maker = DecisionMaker(solver=engine)
        gs = _make_game_state(_cards("Ah As"))
        ctx = GameContext.cash_game(100.0)
        decision = maker.make_decision(gs, ctx, 0)
        # AA should still raise
        assert decision.action in (ActionType.RAISE, ActionType.ALL_IN)
        # Reasoning should mention solver
        assert "Solver" in decision.reasoning or "solver" in decision.reasoning.lower()

    def test_make_decision_detailed(self):
        """make_decision_detailed should return both Decision and SolverResult."""
        engine = SolverEngine()
        maker = DecisionMaker(solver=engine)
        gs = _make_game_state(_cards("Ah Kh"))
        ctx = GameContext.cash_game(100.0)
        decision, solver_result = maker.make_decision_detailed(gs, ctx, 0)
        assert decision is not None
        assert solver_result is not None
        assert solver_result.source in ("preflop_lookup", "heuristic")

    def test_make_decision_detailed_no_solver(self):
        """make_decision_detailed without solver returns None for solver_result."""
        maker = DecisionMaker()
        gs = _make_game_state(_cards("Ah Kh"))
        ctx = GameContext.cash_game(100.0)
        decision, solver_result = maker.make_decision_detailed(gs, ctx, 0)
        assert decision is not None
        assert solver_result is None

    def test_aa_always_raises(self):
        """AA should be a raise/all-in from every position."""
        engine = SolverEngine()
        maker = DecisionMaker(solver=engine)
        for pos in [Position.UTG, Position.MP, Position.CO, Position.BTN, Position.SB]:
            gs = _make_game_state(_cards("Ah As"), position=pos)
            ctx = GameContext.cash_game(100.0)
            decision = maker.make_decision(gs, ctx, 0)
            assert decision.action in (ActionType.RAISE, ActionType.ALL_IN), (
                f"AA should raise from {pos}, got {decision.action}"
            )
