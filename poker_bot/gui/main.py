"""Entry point for the poker coach GUI.

Usage:
    python -m poker_bot.gui.main
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from poker_bot.gui.engine_adapter import EngineAdapter
from poker_bot.gui.main_window import MainWindow
from poker_bot.gui.presenter import GUIPresenter
from poker_bot.gui.styles import APP_STYLESHEET
from poker_bot.interface.opponent_tracker import OpponentTracker


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Poker Coach")
    app.setStyleSheet(APP_STYLESHEET)

    window = MainWindow()
    adapter = EngineAdapter()
    tracker = OpponentTracker()
    presenter = GUIPresenter(view=window, engine=adapter, tracker=tracker)

    # Wire UI signals to presenter
    window.input_panel.solve_requested.connect(presenter.on_solve_clicked)
    window.hud_panel.villain_changed.connect(presenter.on_villain_changed)

    window.show()
    exit_code = app.exec()

    # Cleanup
    adapter.shutdown()
    tracker.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
