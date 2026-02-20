"""Tests for SolverProtocol, parallel Monte Carlo, logging, and resilience."""

import logging
import time
from unittest.mock import MagicMock, patch

from poker_bot.core.equity_calculator import EquityCalculator
from poker_bot.core.game_context import GameContext, TournamentPhase
from poker_bot.core.game_state import GameState, PlayerState
from poker_bot.solver.data_structures import (
    ActionFrequency,
    SolverProtocol,
    SolverResult,
    StrategyNode,
)
from poker_bot.solver.engine import SolverEngine
from poker_bot.strategy.decision_maker import DecisionMaker, PriorAction
from poker_bot.strategy.preflop_ranges import Range
from poker_bot.utils.card import Card
from poker_bot.utils.constants import Action, Position, Rank, Street, Suit


def _card(s: str) -> Card:
    return Card.from_str(s)


def _cards(s: str) -> list[Card]:
    return [Card.from_str(c) for c in s.split()]


def _make_game_state(hero_cards, position=Position.BTN, stack=100.0,
                     pot=3.0, street=Street.PREFLOP, community_cards=None):
    hero = PlayerState(name="Hero", chips=stack, position=position,
                       hole_cards=hero_cards, is_active=True)
    villain = PlayerState(name="Villain", chips=stack, position=Position.UTG,
                          hole_cards=[], is_active=True)
    return GameState(players=[hero, villain], small_blind=0.5, big_blind=1.0,
                     pot=pot, current_street=street,
                     community_cards=community_cards or [])


# ---------------------------------------------------------------------------
# Task 1: SolverProtocol
# ---------------------------------------------------------------------------


class TestSolverProtocol:
    def test_solver_engine_satisfies_protocol(self):
        """SolverEngine should be recognized as implementing SolverProtocol."""
        engine = SolverEngine()
        assert isinstance(engine, SolverProtocol)

    def test_custom_solver_satisfies_protocol(self):
        """A custom class with the right solve() signature satisfies the protocol."""
        class MockSolver:
            def solve(self, game_state, context, hero_index,
                      action_history=None, opponent_range=None):
                return SolverResult(
                    strategy=StrategyNode(actions=[
                        ActionFrequency("fold", 1.0),
                    ]),
                    source="mock",
                    confidence=0.9,
                )

        solver = MockSolver()
        assert isinstance(solver, SolverProtocol)

    def test_decision_maker_accepts_custom_solver(self):
        """DecisionMaker should work with any SolverProtocol implementation."""
        class AlwaysRaiseSolver:
            def solve(self, game_state, context, hero_index,
                      action_history=None, opponent_range=None):
                return SolverResult(
                    strategy=StrategyNode(actions=[
                        ActionFrequency("raise", 1.0, 6.0, 1.0),
                    ]),
                    source="custom",
                    confidence=0.95,
                )

        maker = DecisionMaker(solver=AlwaysRaiseSolver())
        gs = _make_game_state(_cards("Ah Kh"))
        ctx = GameContext.cash_game(100.0)
        decision = maker.make_decision(gs, ctx, 0)
        assert "custom" in decision.reasoning

    def test_non_solver_does_not_satisfy_protocol(self):
        """An object without solve() should not satisfy SolverProtocol."""
        class NotASolver:
            def compute(self):
                pass

        assert not isinstance(NotASolver(), SolverProtocol)


# ---------------------------------------------------------------------------
# Task 2: Parallel Monte Carlo
# ---------------------------------------------------------------------------


class TestParallelMonteCarlo:
    def test_parallel_produces_valid_equity(self):
        """Parallel MC should produce equity in [0, 1]."""
        hand = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.SPADES)]
        opp = Range().add("KK,QQ,JJ")
        board = [Card(Rank.SEVEN, Suit.CLUBS), Card(Rank.THREE, Suit.HEARTS),
                 Card(Rank.NINE, Suit.DIAMONDS)]

        result = EquityCalculator.parallel_hand_vs_range(
            hand, opp, board, simulations=1000,
        )
        assert 0.0 <= result.equity <= 1.0
        assert result.simulations == 1000

    def test_parallel_falls_back_for_small_sims(self):
        """Under 500 sims, parallel should delegate to sequential."""
        hand = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.SPADES)]
        opp = Range().add("QQ,JJ")
        board = [Card(Rank.SEVEN, Suit.CLUBS), Card(Rank.THREE, Suit.HEARTS),
                 Card(Rank.NINE, Suit.DIAMONDS)]

        with patch.object(EquityCalculator, 'hand_vs_range',
                          wraps=EquityCalculator.hand_vs_range) as mock:
            EquityCalculator.parallel_hand_vs_range(
                hand, opp, board, simulations=200,
            )
            mock.assert_called_once()

    def test_parallel_no_valid_combos_raises(self):
        """Should raise ValueError if no valid combos exist."""
        hand = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.SPADES)]
        opp = Range()  # Empty range

        try:
            EquityCalculator.parallel_hand_vs_range(
                hand, opp, simulations=1000,
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_parallel_consistent_with_sequential(self):
        """Parallel and sequential should produce similar equity (within noise)."""
        hand = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.SPADES)]
        opp = Range().add("KK,QQ,JJ,TT,AKs")
        board = [Card(Rank.KING, Suit.DIAMONDS), Card(Rank.SEVEN, Suit.CLUBS),
                 Card(Rank.THREE, Suit.HEARTS)]

        r_seq = EquityCalculator.hand_vs_range(hand, opp, board, simulations=3000)
        r_par = EquityCalculator.parallel_hand_vs_range(hand, opp, board, simulations=3000)

        # Both should show AA ahead vs this range on K73 board
        # Allow reasonable MC variance (within 10%)
        assert abs(r_seq.equity - r_par.equity) < 0.10


# ---------------------------------------------------------------------------
# Task 3: Structured Logging
# ---------------------------------------------------------------------------


class TestStructuredLogging:
    def test_solve_logs_info(self, caplog):
        """solve() should log an INFO message with timing and recommendation."""
        engine = SolverEngine()
        gs = _make_game_state(_cards("Ah Kh"))
        ctx = GameContext.cash_game(100.0)

        with caplog.at_level(logging.INFO, logger="poker_bot.solver"):
            engine.solve(gs, ctx, 0)

        assert len(caplog.records) >= 1
        record = caplog.records[-1]
        assert record.levelname == "INFO"
        # Should contain timing (ms), source, confidence
        assert "ms" in record.message
        assert "source=" in record.message
        assert "confidence=" in record.message

    def test_solve_logs_debug_for_phases(self, caplog):
        """solve() should log DEBUG for individual phases."""
        engine = SolverEngine()
        gs = _make_game_state(_cards("Ah Kh"))
        ctx = GameContext.cash_game(100.0)

        with caplog.at_level(logging.DEBUG, logger="poker_bot.solver"):
            engine.solve(gs, ctx, 0)

        messages = [r.message for r in caplog.records]
        # Should have at least one debug message about preflop lookup
        assert any("Preflop lookup" in m for m in messages)

    def test_tournament_logs_icm_info(self, caplog):
        """Tournament solve should log ICM details at DEBUG level."""
        engine = SolverEngine()
        gs = _make_game_state(_cards("8h 7h"))
        ctx = GameContext.tournament(
            stack_bb=30.0, phase=TournamentPhase.BUBBLE,
            players_remaining=20,
        )

        with caplog.at_level(logging.DEBUG, logger="poker_bot.solver"):
            engine.solve(gs, ctx, 0)

        messages = [r.message for r in caplog.records]
        assert any("Tournament mode" in m for m in messages)
        assert any("ICM adjustment" in m for m in messages)


# ---------------------------------------------------------------------------
# Task 4: Resilience
# ---------------------------------------------------------------------------


class TestResilience:
    def test_corrupted_solver_falls_back(self, caplog):
        """If internal solver raises, solve() returns safe fallback."""
        engine = SolverEngine()
        # Corrupt the preflop solver's data to trigger an error
        engine._preflop._json_data = "not a dict"
        engine._preflop._db = None

        gs = _make_game_state(_cards("Ah Kh"))
        ctx = GameContext.cash_game(100.0)

        with caplog.at_level(logging.ERROR, logger="poker_bot.solver"):
            result = engine.solve(gs, ctx, 0)

        # Should return fold with zero confidence (triggers heuristic fallback)
        assert result.confidence == 0.0
        assert result.source == "fallback"
        assert result.strategy.recommended_action.action == "fold"

        # Should have logged the error
        assert any(r.levelname == "ERROR" for r in caplog.records)

    def test_fallback_has_zero_confidence(self):
        """Fallback result should have confidence 0 so DecisionMaker uses heuristic."""
        engine = SolverEngine()
        engine._preflop._json_data = "broken"
        engine._preflop._db = None

        gs = _make_game_state(_cards("Ah As"))
        ctx = GameContext.cash_game(100.0)
        result = engine.solve(gs, ctx, 0)

        # DecisionMaker should ignore this (confidence < 0.5)
        assert result.confidence < 0.5

    def test_decision_maker_uses_heuristic_on_solver_failure(self):
        """DecisionMaker should fall back to heuristic when solver fails."""
        engine = SolverEngine()
        engine._preflop._json_data = "broken"
        engine._preflop._db = None

        maker = DecisionMaker(solver=engine)
        gs = _make_game_state(_cards("Ah As"))
        ctx = GameContext.cash_game(100.0)
        decision = maker.make_decision(gs, ctx, 0)

        # AA should still raise via heuristic fallback
        assert decision.action.value in ("RAISE", "ALL_IN")
        # Reasoning should NOT mention solver (fell back to heuristic)
        assert "Solver" not in decision.reasoning
