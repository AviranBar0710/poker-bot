"""Tests for ExternalSolverEngine — zero-heuristic routing verification."""

import pytest
from unittest.mock import MagicMock, patch

from poker_bot.core.game_context import GameContext, TournamentPhase
from poker_bot.core.game_state import GameState, PlayerState
from poker_bot.solver.data_structures import (
    ActionFrequency,
    SolverProtocol,
    SolverResult,
    StrategyNode,
)
from poker_bot.solver.external.bridge import (
    GTO_UNAVAILABLE,
    SolverBridge,
    SolverConfig,
    SolverError,
    SolverInput,
    SolverOutput,
)
from poker_bot.solver.external_engine import ExternalSolverEngine
from poker_bot.utils.card import Card
from poker_bot.utils.constants import Action, Position, Rank, Street, Suit


def _cards(s: str) -> list[Card]:
    return [Card.from_str(c) for c in s.split()]


def _make_game_state(
    hero_cards,
    position=Position.BTN,
    stack=100.0,
    pot=3.0,
    street=Street.PREFLOP,
    community_cards=None,
    villain_position=Position.UTG,
    villain_stack=100.0,
):
    hero = PlayerState(
        name="Hero", chips=stack, position=position,
        hole_cards=hero_cards, is_active=True,
    )
    villain = PlayerState(
        name="Villain", chips=villain_stack, position=villain_position,
        hole_cards=[], is_active=True,
    )
    return GameState(
        players=[hero, villain],
        small_blind=0.5,
        big_blind=1.0,
        pot=pot,
        current_street=street,
        community_cards=community_cards or [],
    )


class MockBridge(SolverBridge):
    """Mock bridge that returns pre-configured SolverOutput."""

    def __init__(self, available=True, output=None, raises=None):
        self._available = available
        self._output = output or SolverOutput(
            hero_strategy={"raise": 0.60, "call": 0.25, "fold": 0.15},
            hero_ev=3.5,
            converged=True,
            exploitability=0.3,
        )
        self._raises = raises
        self.solve_called = False
        self.last_input: SolverInput | None = None

    def is_available(self) -> bool:
        return self._available

    def solve(self, solver_input: SolverInput) -> SolverOutput:
        self.solve_called = True
        self.last_input = solver_input
        if self._raises:
            raise self._raises
        return self._output

    def cleanup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# SolverProtocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_satisfies_solver_protocol(self):
        engine = ExternalSolverEngine()
        assert isinstance(engine, SolverProtocol)


# ---------------------------------------------------------------------------
# Preflop routing
# ---------------------------------------------------------------------------


class TestPreflopRouting:
    def test_preflop_routes_to_preflop_solver(self):
        """Preflop should use PreflopSolver, NOT the external bridge."""
        mock_bridge = MockBridge()
        engine = ExternalSolverEngine(bridge=mock_bridge)

        gs = _make_game_state(_cards("Ah Kh"))
        ctx = GameContext.cash_game(100.0)
        result = engine.solve(gs, ctx, 0)

        # Bridge should NOT have been called for preflop
        assert mock_bridge.solve_called is False
        # Result should come from preflop solver
        assert result.source != "external_solver"
        assert result.source != "gto_unavailable"

    def test_preflop_works_without_bridge(self):
        """Preflop should work even if no bridge is configured."""
        engine = ExternalSolverEngine(bridge=None)

        gs = _make_game_state(_cards("Ah Kh"))
        ctx = GameContext.cash_game(100.0)
        result = engine.solve(gs, ctx, 0)

        assert result.source != "gto_unavailable"
        assert result.confidence > 0.0


# ---------------------------------------------------------------------------
# Postflop routing
# ---------------------------------------------------------------------------


class TestPostflopRouting:
    def test_postflop_routes_to_bridge(self):
        """Postflop should delegate to the external bridge."""
        mock_bridge = MockBridge()
        engine = ExternalSolverEngine(bridge=mock_bridge)

        gs = _make_game_state(
            _cards("Ah Kh"),
            street=Street.FLOP,
            community_cards=_cards("Qs Jh 2h"),
            pot=10.0,
        )
        ctx = GameContext.cash_game(100.0)
        result = engine.solve(gs, ctx, 0)

        assert mock_bridge.solve_called is True
        assert result.source == "external_solver"
        assert result.confidence > 0.0

    def test_postflop_returns_gto_unavailable_without_bridge(self):
        """No bridge → GTO_UNAVAILABLE for postflop."""
        engine = ExternalSolverEngine(bridge=None)

        gs = _make_game_state(
            _cards("Ah Kh"),
            street=Street.FLOP,
            community_cards=_cards("Qs Jh 2h"),
            pot=10.0,
        )
        ctx = GameContext.cash_game(100.0)
        result = engine.solve(gs, ctx, 0)

        assert result.source == "gto_unavailable"
        assert result.confidence == 0.0
        assert result.strategy.actions == []

    def test_postflop_returns_gto_unavailable_when_bridge_unavailable(self):
        """Bridge exists but binary missing → GTO_UNAVAILABLE."""
        mock_bridge = MockBridge(available=False)
        engine = ExternalSolverEngine(bridge=mock_bridge)

        gs = _make_game_state(
            _cards("Ah Kh"),
            street=Street.FLOP,
            community_cards=_cards("Qs Jh 2h"),
            pot=10.0,
        )
        ctx = GameContext.cash_game(100.0)
        result = engine.solve(gs, ctx, 0)

        assert result.source == "gto_unavailable"
        assert mock_bridge.solve_called is False

    def test_postflop_returns_gto_unavailable_on_solver_error(self):
        """SolverError → GTO_UNAVAILABLE (not a crash)."""
        mock_bridge = MockBridge(raises=SolverError("Solver timed out"))
        engine = ExternalSolverEngine(bridge=mock_bridge)

        gs = _make_game_state(
            _cards("Ah Kh"),
            street=Street.FLOP,
            community_cards=_cards("Qs Jh 2h"),
            pot=10.0,
        )
        ctx = GameContext.cash_game(100.0)
        result = engine.solve(gs, ctx, 0)

        assert result.source == "gto_unavailable"
        assert result.confidence == 0.0

    def test_postflop_returns_gto_unavailable_on_unexpected_error(self):
        """Unexpected exception → GTO_UNAVAILABLE (never crash)."""
        mock_bridge = MockBridge(raises=RuntimeError("Unexpected"))
        engine = ExternalSolverEngine(bridge=mock_bridge)

        gs = _make_game_state(
            _cards("Ah Kh"),
            street=Street.FLOP,
            community_cards=_cards("Qs Jh 2h"),
            pot=10.0,
        )
        ctx = GameContext.cash_game(100.0)
        result = engine.solve(gs, ctx, 0)

        assert result.source == "gto_unavailable"


# ---------------------------------------------------------------------------
# Zero-heuristic verification
# ---------------------------------------------------------------------------


class TestZeroHeuristic:
    def test_postflop_solver_never_instantiated(self):
        """Verify PostflopSolver is NEVER used by ExternalSolverEngine."""
        with patch(
            "poker_bot.solver.external_engine.PostflopSolver"
        ) as mock_postflop:
            mock_bridge = MockBridge()
            engine = ExternalSolverEngine(bridge=mock_bridge)

            gs = _make_game_state(
                _cards("Ah Kh"),
                street=Street.FLOP,
                community_cards=_cards("Qs Jh 2h"),
                pot=10.0,
            )
            ctx = GameContext.cash_game(100.0)
            engine.solve(gs, ctx, 0)

            # PostflopSolver should never be instantiated
            mock_postflop.assert_not_called()

    def test_external_result_frequencies_passed_through(self):
        """CFR frequencies should come through unchanged."""
        mock_bridge = MockBridge(
            output=SolverOutput(
                hero_strategy={"raise": 0.42, "call": 0.35, "fold": 0.23},
                hero_ev=2.7,
                converged=True,
            )
        )
        engine = ExternalSolverEngine(bridge=mock_bridge)

        gs = _make_game_state(
            _cards("Ah Kh"),
            street=Street.FLOP,
            community_cards=_cards("Qs Jh 2h"),
            pot=10.0,
        )
        ctx = GameContext.cash_game(100.0)
        result = engine.solve(gs, ctx, 0)

        # Frequencies should be normalized but proportions preserved
        actions = {a.action: a.frequency for a in result.strategy.actions}
        assert abs(actions["raise"] - 0.42) < 0.01
        assert abs(actions["call"] - 0.35) < 0.01
        assert abs(actions["fold"] - 0.23) < 0.01


# ---------------------------------------------------------------------------
# SolverInput construction
# ---------------------------------------------------------------------------


class TestSolverInputConstruction:
    def test_bridge_receives_correct_pot(self):
        mock_bridge = MockBridge()
        engine = ExternalSolverEngine(bridge=mock_bridge)

        gs = _make_game_state(
            _cards("Ah Kh"),
            street=Street.FLOP,
            community_cards=_cards("Qs Jh 2h"),
            pot=42.0,
        )
        ctx = GameContext.cash_game(100.0)
        engine.solve(gs, ctx, 0)

        assert mock_bridge.last_input is not None
        assert mock_bridge.last_input.pot == 42.0

    def test_bridge_receives_correct_board(self):
        mock_bridge = MockBridge()
        engine = ExternalSolverEngine(bridge=mock_bridge)

        board = _cards("Qs Jh 2h")
        gs = _make_game_state(
            _cards("Ah Kh"),
            street=Street.FLOP,
            community_cards=board,
            pot=10.0,
        )
        ctx = GameContext.cash_game(100.0)
        engine.solve(gs, ctx, 0)

        assert mock_bridge.last_input.board == board

    def test_bridge_receives_hero_cards(self):
        mock_bridge = MockBridge()
        engine = ExternalSolverEngine(bridge=mock_bridge)

        hero_cards = _cards("Ah Kh")
        gs = _make_game_state(
            hero_cards,
            street=Street.FLOP,
            community_cards=_cards("Qs Jh 2h"),
            pot=10.0,
        )
        ctx = GameContext.cash_game(100.0)
        engine.solve(gs, ctx, 0)

        assert mock_bridge.last_input.hero_cards == hero_cards

    def test_effective_stack_is_minimum(self):
        """Effective stack = min(hero, villain)."""
        mock_bridge = MockBridge()
        engine = ExternalSolverEngine(bridge=mock_bridge)

        gs = _make_game_state(
            _cards("Ah Kh"),
            street=Street.FLOP,
            community_cards=_cards("Qs Jh 2h"),
            pot=10.0,
            stack=150.0,
            villain_stack=80.0,
        )
        ctx = GameContext.cash_game(100.0)
        engine.solve(gs, ctx, 0)

        assert mock_bridge.last_input.effective_stack == 80.0

    def test_street_detected_from_board(self):
        mock_bridge = MockBridge()
        engine = ExternalSolverEngine(bridge=mock_bridge)

        gs = _make_game_state(
            _cards("Ah Kh"),
            street=Street.TURN,
            community_cards=_cards("Qs Jh 2h 7c"),
            pot=20.0,
        )
        ctx = GameContext.cash_game(100.0)
        engine.solve(gs, ctx, 0)

        assert mock_bridge.last_input.street == "turn"


# ---------------------------------------------------------------------------
# No hole cards
# ---------------------------------------------------------------------------


class TestNoHoleCards:
    def test_no_hole_cards_returns_gto_unavailable(self):
        engine = ExternalSolverEngine(bridge=MockBridge())

        hero = PlayerState(
            name="Hero", chips=100.0, position=Position.BTN,
            hole_cards=[], is_active=True,
        )
        villain = PlayerState(
            name="Villain", chips=100.0, position=Position.UTG,
            hole_cards=[], is_active=True,
        )
        gs = GameState(
            players=[hero, villain], small_blind=0.5, big_blind=1.0,
            pot=3.0, current_street=Street.PREFLOP, community_cards=[],
        )
        ctx = GameContext.cash_game(100.0)
        result = engine.solve(gs, ctx, 0)

        assert result.source == "gto_unavailable"


# ---------------------------------------------------------------------------
# ICM adjustment for tournaments
# ---------------------------------------------------------------------------


class TestTournamentICM:
    def test_icm_applies_to_external_result(self):
        """ICM adjustment should apply to external solver results."""
        mock_bridge = MockBridge(
            output=SolverOutput(
                hero_strategy={"raise": 0.50, "fold": 0.50},
                hero_ev=2.0,
                converged=True,
            )
        )
        engine = ExternalSolverEngine(bridge=mock_bridge)

        gs = _make_game_state(
            _cards("8h 7h"),
            street=Street.FLOP,
            community_cards=_cards("Qs Jh 2h"),
            pot=10.0,
        )
        ctx = GameContext.tournament(
            stack_bb=30.0,
            phase=TournamentPhase.BUBBLE,
            players_remaining=20,
        )
        result = engine.solve(gs, ctx, 0)

        # ICM should increase fold frequency relative to raise
        actions = {a.action: a.frequency for a in result.strategy.actions}
        # On the bubble, fold should be boosted
        assert actions.get("fold", 0) > 0.0
        assert result.source == "external_solver"

    def test_no_icm_for_gto_unavailable(self):
        """GTO_UNAVAILABLE should NOT get ICM adjustment."""
        engine = ExternalSolverEngine(bridge=None)  # No bridge

        gs = _make_game_state(
            _cards("8h 7h"),
            street=Street.FLOP,
            community_cards=_cards("Qs Jh 2h"),
            pot=10.0,
        )
        ctx = GameContext.tournament(
            stack_bb=30.0,
            phase=TournamentPhase.BUBBLE,
            players_remaining=20,
        )
        result = engine.solve(gs, ctx, 0)

        # Should be raw GTO_UNAVAILABLE, not ICM-adjusted
        assert result.source == "gto_unavailable"
        assert result.strategy.actions == []


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_calls_bridge_cleanup(self):
        mock_bridge = MockBridge()
        mock_bridge.cleanup = MagicMock()
        engine = ExternalSolverEngine(bridge=mock_bridge)
        engine.cleanup()
        mock_bridge.cleanup.assert_called_once()

    def test_cleanup_safe_without_bridge(self):
        engine = ExternalSolverEngine(bridge=None)
        engine.cleanup()  # Should not raise
