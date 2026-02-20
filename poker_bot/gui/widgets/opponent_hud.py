"""Opponent HUD panel widget.

Displays opponent stats (VPIP, PFR, 3-bet, AF, fold-to-cbet),
player type classification, and timestamped notes. The villain
name input has a 300ms debounce timer for auto-lookup.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from poker_bot.interface.opponent_tracker import OpponentStats


class OpponentHudPanel(QWidget):
    """Bottom panel: villain name input with debounced lookup + stats display."""

    villain_changed = Signal(str)  # Emitted after 300ms debounce

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)
        self._debounce_timer.timeout.connect(self._emit_villain)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        # Villain name input row
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Villain:"))
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("Enter opponent name...")
        self._name_input.textChanged.connect(self._on_text_changed)
        name_row.addWidget(self._name_input)
        layout.addLayout(name_row)

        # Stats display
        self._stats_group = QGroupBox("Opponent Stats")
        stats_layout = QVBoxLayout(self._stats_group)

        self._type_label = QLabel("Type: —")
        self._type_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        stats_layout.addWidget(self._type_label)

        self._stats_label = QLabel("")
        self._stats_label.setWordWrap(True)
        self._stats_label.setStyleSheet("font-size: 12px; color: #374151;")
        stats_layout.addWidget(self._stats_label)

        self._advice_label = QLabel("")
        self._advice_label.setWordWrap(True)
        self._advice_label.setStyleSheet(
            "font-size: 12px; color: #1d4ed8; font-style: italic; padding-top: 4px;"
        )
        stats_layout.addWidget(self._advice_label)

        # Notes
        self._notes_label = QLabel("Notes:")
        self._notes_label.setStyleSheet("font-weight: bold; font-size: 12px; padding-top: 4px;")
        stats_layout.addWidget(self._notes_label)

        self._notes_text = QTextEdit()
        self._notes_text.setReadOnly(True)
        self._notes_text.setMaximumHeight(80)
        self._notes_text.setStyleSheet("font-size: 11px; background: #f9fafb;")
        stats_layout.addWidget(self._notes_text)

        layout.addWidget(self._stats_group)

    def _on_text_changed(self, text: str) -> None:
        self._debounce_timer.start()

    def _emit_villain(self) -> None:
        self.villain_changed.emit(self._name_input.text())

    def get_villain_name(self) -> str:
        return self._name_input.text().strip()

    def show_stats(
        self,
        stats: OpponentStats | None,
        notes: list[tuple[str, str]],
        advice: str,
    ) -> None:
        """Update the HUD display with opponent data."""
        if stats is None:
            self._type_label.setText("Type: —")
            self._stats_label.setText("No data for this player.")
            self._advice_label.clear()
            self._notes_text.clear()
            return

        self._type_label.setText(f"Type: {stats.player_type}")

        if stats.hands_seen > 0:
            stat_parts = [
                f"VPIP: {stats.vpip_pct:.0f}%",
                f"PFR: {stats.pfr_pct:.0f}%",
                f"3-Bet: {stats.three_bet_pct:.0f}%",
                f"AF: {stats.aggression_factor:.1f}",
                f"FtCB: {stats.fold_to_cbet_pct:.0f}%",
                f"Hands: {stats.hands_seen}",
            ]
            self._stats_label.setText(" | ".join(stat_parts))
        else:
            self._stats_label.setText("No stats yet.")

        self._advice_label.setText(advice)

        if notes:
            lines = [f"{i+1}. {note} ({ts})" for i, (note, ts) in enumerate(notes)]
            self._notes_text.setPlainText("\n".join(lines))
        else:
            self._notes_text.clear()
