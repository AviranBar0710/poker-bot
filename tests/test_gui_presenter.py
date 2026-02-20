"""Tests for GUIPresenter with mock PokerView.

These tests verify the presenter logic without any Qt/PySide6 dependency
by mocking both the view and the engine adapter.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from poker_bot.gui.presenter import GUIPresenter
from poker_bot.solver.data_structures import (
    ActionFrequency,
    SolverResult,
    StrategyNode,
)
from poker_bot.strategy.decision_maker import ActionType, Decision


def _make_mock_view(**overrides):
    """Create a mock PokerView with sensible defaults."""
    view = MagicMock()
    view.get_hero_cards.return_value = overrides.get("hero_cards", ["Ah", "Ks"])
    view.get_board_cards.return_value = overrides.get("board_cards", [])
    view.get_pot_bb.return_value = overrides.get("pot_bb", 3.0)
    view.get_bet_bb.return_value = overrides.get("bet_bb", 2.0)
    view.get_stack_bb.return_value = overrides.get("stack_bb", 50.0)
    view.get_num_opponents.return_value = overrides.get("num_opponents", 2)
    view.get_street.return_value = overrides.get("street", "PREFLOP")
    view.get_position.return_value = overrides.get("position", "BTN")
    view.get_position_ip.return_value = overrides.get("position_ip", True)
    view.get_game_type.return_value = overrides.get("game_type", "Cash")
    view.get_villain_name.return_value = overrides.get("villain_name", "")
    return view


def _make_mock_engine():
    """Create a mock EngineAdapter."""
    engine = MagicMock()
    engine.solving_started = MagicMock()
    engine.solving_finished = MagicMock()
    engine.solving_error = MagicMock()
    # Make connect() callable
    engine.solving_started.connect = MagicMock()
    engine.solving_finished.connect = MagicMock()
    engine.solving_error.connect = MagicMock()
    return engine


class TestPresenterSolveClicked:
    """Tests for on_solve_clicked()."""

    def test_solve_dispatches_request(self):
        view = _make_mock_view()
        engine = _make_mock_engine()
        presenter = GUIPresenter(view=view, engine=engine)

        presenter.on_solve_clicked()

        engine.request_solve.assert_called_once()
        request = engine.request_solve.call_args[0][0]
        assert len(request.game_state.players) == 3
        assert request.game_state.players[0].name == "Hero"

    def test_solve_shows_error_on_missing_cards(self):
        view = _make_mock_view(hero_cards=["Ah", ""])
        engine = _make_mock_engine()
        presenter = GUIPresenter(view=view, engine=engine)

        presenter.on_solve_clicked()

        view.show_error.assert_called_once()
        engine.request_solve.assert_not_called()

    def test_solve_shows_error_on_one_card(self):
        view = _make_mock_view(hero_cards=["Ah"])
        engine = _make_mock_engine()
        presenter = GUIPresenter(view=view, engine=engine)

        presenter.on_solve_clicked()

        view.show_error.assert_called_once()

    def test_solve_with_board_cards(self):
        view = _make_mock_view(
            street="FLOP",
            board_cards=["Jh", "8d", "3c"],
        )
        engine = _make_mock_engine()
        presenter = GUIPresenter(view=view, engine=engine)

        presenter.on_solve_clicked()

        engine.request_solve.assert_called_once()
        request = engine.request_solve.call_args[0][0]
        assert len(request.game_state.community_cards) == 3

    def test_solve_with_tournament(self):
        view = _make_mock_view(game_type="Tournament")
        engine = _make_mock_engine()
        presenter = GUIPresenter(view=view, engine=engine)

        presenter.on_solve_clicked()

        request = engine.request_solve.call_args[0][0]
        assert request.context.is_tournament


class TestPresenterCallbacks:
    """Tests for engine callback handling."""

    def test_solving_started_shows_indicator(self):
        view = _make_mock_view()
        engine = _make_mock_engine()
        presenter = GUIPresenter(view=view, engine=engine)

        # Simulate the engine calling back
        presenter._on_solving_started()
        view.show_solving.assert_called_once()

    def test_solving_finished_shows_result(self):
        view = _make_mock_view()
        engine = _make_mock_engine()
        presenter = GUIPresenter(view=view, engine=engine)

        decision = Decision(
            action=ActionType.RAISE,
            amount=6.5,
            reasoning="Open raise with AKs",
            equity=0.65,
            pot_odds=0.30,
        )
        solver_result = SolverResult(
            strategy=StrategyNode(actions=[
                ActionFrequency("raise", 0.62, 6.5),
                ActionFrequency("call", 0.25, 2.0),
                ActionFrequency("fold", 0.13),
            ]),
            source="preflop_lookup",
            confidence=0.95,
            ev=2.35,
        )

        # Create a mock SolveResponse
        response = MagicMock()
        response.decision = decision
        response.solver_result = solver_result

        presenter._on_solving_finished(response)
        view.show_result.assert_called_once()
        assert view.show_result.call_args[0][0] is decision

    def test_solving_error_shows_message(self):
        view = _make_mock_view()
        engine = _make_mock_engine()
        presenter = GUIPresenter(view=view, engine=engine)

        presenter._on_solving_error("MC calculation failed")
        view.show_error.assert_called_once_with("Solver error: MC calculation failed")


class TestPresenterVillainLookup:
    """Tests for on_villain_changed()."""

    def test_empty_name_clears_stats(self):
        view = _make_mock_view()
        engine = _make_mock_engine()
        presenter = GUIPresenter(view=view, engine=engine)

        presenter.on_villain_changed("")
        view.show_opponent_stats.assert_called_once_with(None, [], "")

    def test_unknown_villain_shows_no_data(self):
        view = _make_mock_view()
        engine = _make_mock_engine()
        presenter = GUIPresenter(view=view, engine=engine)

        presenter.on_villain_changed("UnknownPlayer")
        view.show_opponent_stats.assert_called_once()
        args = view.show_opponent_stats.call_args[0]
        assert args[0] is None  # stats
