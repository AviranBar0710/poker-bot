"""Abstract view interface for the poker coach GUI.

The PokerView Protocol defines the contract between the GUIPresenter
and any concrete UI framework (PySide6, PyQt6, etc.). The presenter
depends only on this protocol — never on framework-specific imports.
"""

from __future__ import annotations

from typing import Protocol

from poker_bot.interface.opponent_tracker import OpponentStats
from poker_bot.solver.data_structures import SolverResult
from poker_bot.strategy.decision_maker import ActionType, Decision


class PokerView(Protocol):
    """Interface that any GUI framework must implement."""

    # --- Input reading ---

    def get_hero_cards(self) -> list[str]:
        """Return hero's hole cards as 2-char strings, e.g. ['Ah', 'Ks']."""
        ...

    def get_board_cards(self) -> list[str]:
        """Return community cards as 2-char strings."""
        ...

    def get_pot_bb(self) -> float:
        """Return pot size in big blinds."""
        ...

    def get_bet_bb(self) -> float:
        """Return current bet to face in big blinds."""
        ...

    def get_stack_bb(self) -> float:
        """Return hero's stack in big blinds."""
        ...

    def get_num_opponents(self) -> int:
        """Return number of opponents."""
        ...

    def get_street(self) -> str:
        """Return current street as uppercase string."""
        ...

    def get_position_ip(self) -> bool:
        """Return True if hero is in position, False if OOP."""
        ...

    def get_game_type(self) -> str:
        """Return 'cash' or 'tournament'."""
        ...

    def get_position(self) -> str:
        """Return hero's position as uppercase string (BTN, SB, BB, etc.)."""
        ...

    def get_villain_name(self) -> str:
        """Return the villain name from the HUD input."""
        ...

    # --- Output display ---

    def show_solving(self) -> None:
        """Show a 'Calculating...' indicator."""
        ...

    def show_result(
        self,
        decision: Decision,
        solver_result: SolverResult | None,
        hero_cards_str: str,
        opponent_advice: str,
    ) -> None:
        """Display the solver result in the output panel."""
        ...

    def show_error(self, message: str) -> None:
        """Display an error message."""
        ...

    def show_opponent_stats(self, stats: OpponentStats | None, notes: list[tuple[str, str]], advice: str) -> None:
        """Update the opponent HUD panel with stats and notes."""
        ...

    def show_gto_unavailable(self) -> None:
        """Display a warning that no GTO solution is available."""
        ...

    def clear_result(self) -> None:
        """Clear the solver output panel."""
        ...
