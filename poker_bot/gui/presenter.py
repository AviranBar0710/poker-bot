"""Framework-agnostic presenter for the poker coach GUI.

GUIPresenter mediates between the PokerView (UI) and the engine
(solver + opponent tracker). It has NO Qt/PySide6 imports — it
depends only on the PokerView Protocol and engine types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poker_bot.interface.opponent_tracker import OpponentTracker
from poker_bot.interface.poker_coach import (
    _hand_display,
    _parse_action_history,
    _parse_cards,
)
from poker_bot.interface.situation_builder import build_game_objects
from poker_bot.solver.data_structures import SolverResult
from poker_bot.strategy.decision_maker import Decision, PriorAction
from poker_bot.utils.constants import Position, Street

if TYPE_CHECKING:
    from poker_bot.gui.engine_adapter import EngineAdapter, SolveRequest, SolveResponse
    from poker_bot.gui.view_protocol import PokerView


class GUIPresenter:
    """Coordinates view inputs, engine solving, and result display.

    Framework-agnostic: depends only on the PokerView Protocol.
    """

    def __init__(
        self,
        view: PokerView,
        engine: EngineAdapter,
        tracker: OpponentTracker | None = None,
    ) -> None:
        self._view = view
        self._engine = engine
        self._tracker = tracker or OpponentTracker()

        # Connect engine signals
        self._engine.solving_started.connect(self._on_solving_started)
        self._engine.solving_finished.connect(self._on_solving_finished)
        self._engine.solving_error.connect(self._on_solving_error)

    def on_solve_clicked(self) -> None:
        """Handle the SOLVE button click."""
        from poker_bot.gui.engine_adapter import SolveRequest

        try:
            # Read inputs from the view
            hero_card_strs = self._view.get_hero_cards()
            if len(hero_card_strs) != 2 or not all(hero_card_strs):
                self._view.show_error("Please select both hero cards.")
                return

            hero_cards = _parse_cards(" ".join(hero_card_strs))

            board_card_strs = self._view.get_board_cards()
            community_cards = _parse_cards(" ".join(board_card_strs)) if board_card_strs else []

            pot_bb = self._view.get_pot_bb()
            bet_bb = self._view.get_bet_bb()
            stack_bb = self._view.get_stack_bb()
            num_opponents = self._view.get_num_opponents()
            street = Street(self._view.get_street())
            position = Position(self._view.get_position())
            game_type = self._view.get_game_type()
            is_tournament = game_type.lower().startswith("t")

            gs, ctx = build_game_objects(
                hero_cards=hero_cards,
                position=position,
                stack_bb=stack_bb,
                street=street,
                pot_bb=pot_bb,
                current_bet_bb=bet_bb,
                num_opponents=num_opponents,
                community_cards=community_cards,
                is_tournament=is_tournament,
            )

            # Load opponent stats into context if available
            villain_name = self._view.get_villain_name()
            if villain_name:
                player = self._tracker.get_player(villain_name)
                if player:
                    ctx.opponent_stats = {villain_name: player}

            request = SolveRequest(
                game_state=gs,
                context=ctx,
                hero_index=0,
            )
            self._engine.request_solve(request)

        except (ValueError, KeyError) as e:
            self._view.show_error(str(e))

    def on_villain_changed(self, name: str) -> None:
        """Handle villain name change (debounced from view)."""
        name = name.strip()
        if not name:
            self._view.show_opponent_stats(None, [], "")
            return

        stats = self._tracker.get_player(name)
        notes = self._tracker.get_notes(name) if stats else []
        advice = self._tracker.get_range_adjustment(name) if stats else ""
        self._view.show_opponent_stats(stats, notes, advice)

    def _on_solving_started(self) -> None:
        self._view.show_solving()

    def _on_solving_finished(self, response: SolveResponse) -> None:
        decision = response.decision
        solver_result = response.solver_result

        # Check for GTO_UNAVAILABLE — show warning instead of fake result
        if solver_result is not None and solver_result.source == "gto_unavailable":
            self._view.show_gto_unavailable()
            return

        # Build hero cards display string
        hero_card_strs = self._view.get_hero_cards()
        try:
            hero_cards = _parse_cards(" ".join(hero_card_strs))
            hero_display = _hand_display(hero_cards)
        except (ValueError, IndexError):
            hero_display = " ".join(hero_card_strs)

        # Get opponent advice
        villain_name = self._view.get_villain_name()
        opponent_advice = ""
        if villain_name:
            advice = self._tracker.get_range_adjustment(villain_name)
            player = self._tracker.get_player(villain_name)
            if player:
                opponent_advice = f"{player.summary()}\nAdjustment: {advice}"

        self._view.show_result(decision, solver_result, hero_display, opponent_advice)

    def _on_solving_error(self, message: str) -> None:
        self._view.show_error(f"Solver error: {message}")
