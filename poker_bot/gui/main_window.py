"""Main window assembling all panels, implementing the PokerView protocol."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from poker_bot.gui.widgets.input_panel import InputPanel
from poker_bot.gui.widgets.opponent_hud import OpponentHudPanel
from poker_bot.gui.widgets.solver_output import SolverOutputPanel
from poker_bot.interface.opponent_tracker import OpponentStats
from poker_bot.solver.data_structures import SolverResult
from poker_bot.strategy.decision_maker import Decision


class MainWindow(QMainWindow):
    """Top-level window implementing the PokerView protocol.

    Layout:
      - InputPanel (top ~35%)
      - SolverOutputPanel (middle ~40%)
      - OpponentHudPanel (bottom ~25%)
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Poker Coach")
        self.setMinimumSize(720, 640)
        self.resize(800, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Input panel (top)
        self._input_panel = InputPanel()
        layout.addWidget(self._input_panel)

        # Divider
        layout.addWidget(self._divider())

        # Solver output (middle)
        self._output_panel = SolverOutputPanel()
        layout.addWidget(self._output_panel, stretch=1)

        # Divider
        layout.addWidget(self._divider())

        # Opponent HUD (bottom)
        self._hud_panel = OpponentHudPanel()
        layout.addWidget(self._hud_panel)

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #e5e7eb;")
        return line

    # --- PokerView protocol implementation ---

    def get_hero_cards(self) -> list[str]:
        return self._input_panel.get_hero_cards()

    def get_board_cards(self) -> list[str]:
        return self._input_panel.get_board_cards()

    def get_pot_bb(self) -> float:
        return self._input_panel.get_pot_bb()

    def get_bet_bb(self) -> float:
        return self._input_panel.get_bet_bb()

    def get_stack_bb(self) -> float:
        return self._input_panel.get_stack_bb()

    def get_num_opponents(self) -> int:
        return self._input_panel.get_num_opponents()

    def get_street(self) -> str:
        return self._input_panel.get_street()

    def get_position_ip(self) -> bool:
        return self._input_panel.get_position_ip()

    def get_game_type(self) -> str:
        return self._input_panel.get_game_type()

    def get_position(self) -> str:
        return self._input_panel.get_position()

    def get_villain_name(self) -> str:
        return self._hud_panel.get_villain_name()

    def show_solving(self) -> None:
        self._output_panel.show_solving()

    def show_result(
        self,
        decision: Decision,
        solver_result: SolverResult | None,
        hero_cards_str: str,
        opponent_advice: str,
    ) -> None:
        self._output_panel.show_result(decision, solver_result, hero_cards_str, opponent_advice)

    def show_error(self, message: str) -> None:
        self._output_panel.show_error(message)

    def show_opponent_stats(
        self,
        stats: OpponentStats | None,
        notes: list[tuple[str, str]],
        advice: str,
    ) -> None:
        self._hud_panel.show_stats(stats, notes, advice)

    def clear_result(self) -> None:
        self._output_panel.clear()

    # --- Signal accessors for wiring ---

    @property
    def input_panel(self) -> InputPanel:
        return self._input_panel

    @property
    def hud_panel(self) -> OpponentHudPanel:
        return self._hud_panel
