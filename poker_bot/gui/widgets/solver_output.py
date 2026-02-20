"""Solver output panel: action banner, strategy bars, and analysis text.

Displays the recommended action prominently, shows GTO mixed strategy
frequencies as colored horizontal bars, and provides equity/EV/reasoning.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from poker_bot.solver.data_structures import SolverResult
from poker_bot.strategy.decision_maker import ActionType, Decision

# Action → (background color, text color)
_ACTION_COLORS = {
    ActionType.RAISE: ("#22c55e", "#fff"),
    ActionType.CALL: ("#3b82f6", "#fff"),
    ActionType.FOLD: ("#ef4444", "#fff"),
    ActionType.ALL_IN: ("#f97316", "#fff"),
    ActionType.CHECK: ("#6b7280", "#fff"),
    ActionType.LIMP: ("#8b5cf6", "#fff"),
}

_BAR_COLORS = {
    "raise": "#22c55e",
    "call": "#3b82f6",
    "fold": "#ef4444",
    "check": "#6b7280",
    "all_in": "#f97316",
    "allin": "#f97316",
    "limp": "#8b5cf6",
}


class StrategyBarChart(QWidget):
    """Custom-painted horizontal bar chart for GTO mixed strategy frequencies."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._actions: list[tuple[str, float, float]] = []  # (name, freq, amount)
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def set_strategy(self, actions: list[tuple[str, float, float]]) -> None:
        """Set the actions to display: list of (action_name, frequency, amount)."""
        self._actions = sorted(actions, key=lambda a: -a[1])
        self.update()

    def clear(self) -> None:
        self._actions = []
        self.update()

    def paintEvent(self, event) -> None:
        if not self._actions:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        row_h = 24
        margin_left = 80
        bar_max_w = w - margin_left - 120
        y = 4

        label_font = QFont("Segoe UI", 11, QFont.Bold)
        detail_font = QFont("Segoe UI", 10)

        for action_name, freq, amount in self._actions:
            if freq < 0.01:
                continue

            color_hex = _BAR_COLORS.get(action_name.lower(), "#9ca3af")
            bar_color = QColor(color_hex)

            # Action label
            painter.setFont(label_font)
            painter.setPen(QColor("#1f2937"))
            label = action_name.upper()
            painter.drawText(QRectF(4, y, margin_left - 8, row_h), Qt.AlignVCenter | Qt.AlignRight, label)

            # Bar
            bar_w = max(4, int(bar_max_w * freq))
            painter.setBrush(bar_color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(QRectF(margin_left, y + 2, bar_w, row_h - 4), 3, 3)

            # Percentage + amount
            painter.setFont(detail_font)
            painter.setPen(QColor("#374151"))
            detail = f"{freq:.1%}"
            if amount > 0:
                detail += f" ({amount:.1f}bb)"
            painter.drawText(
                QRectF(margin_left + bar_w + 6, y, 110, row_h),
                Qt.AlignVCenter | Qt.AlignLeft,
                detail,
            )

            y += row_h + 2

        painter.end()
        self.setMinimumHeight(y + 4)


class SolverOutputPanel(QWidget):
    """Displays solver results: action banner, strategy bars, analysis."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        # Action banner
        self._banner = QLabel()
        self._banner.setAlignment(Qt.AlignCenter)
        self._banner.setFixedHeight(44)
        self._banner.setStyleSheet(
            "QLabel { font-size: 18px; font-weight: bold; color: #6b7280;"
            " background: #f3f4f6; border-radius: 8px; }"
        )
        self._banner.setText("Click SOLVE to get a recommendation")
        layout.addWidget(self._banner)

        # Strategy bars
        self._bars = StrategyBarChart()
        layout.addWidget(self._bars)

        # Analysis text
        self._analysis = QLabel()
        self._analysis.setWordWrap(True)
        self._analysis.setStyleSheet(
            "QLabel { color: #374151; font-size: 12px; padding: 4px; }"
        )
        layout.addWidget(self._analysis)

        layout.addStretch()

    def show_solving(self) -> None:
        """Show a calculating indicator."""
        self._banner.setText("Calculating...")
        self._banner.setStyleSheet(
            "QLabel { font-size: 18px; font-weight: bold; color: #f59e0b;"
            " background: #fffbeb; border-radius: 8px; }"
        )
        self._bars.clear()
        self._analysis.clear()

    def show_result(
        self,
        decision: Decision,
        solver_result: SolverResult | None,
        hero_cards_str: str,
        opponent_advice: str,
    ) -> None:
        """Display solver result."""
        # Action banner
        action = decision.action
        if action == ActionType.RAISE:
            action_text = f"RAISE to {decision.amount:.1f} bb"
        elif action == ActionType.CALL:
            action_text = f"CALL {decision.amount:.1f} bb"
        elif action == ActionType.ALL_IN:
            action_text = f"ALL-IN ({decision.amount:.1f} bb)"
        elif action == ActionType.LIMP:
            action_text = "LIMP (call 1 bb)"
        else:
            action_text = action.value

        bg, fg = _ACTION_COLORS.get(action, ("#6b7280", "#fff"))
        self._banner.setText(f">>> {action_text} <<<")
        self._banner.setStyleSheet(
            f"QLabel {{ font-size: 18px; font-weight: bold; color: {fg};"
            f" background: {bg}; border-radius: 8px; }}"
        )

        # Strategy bars
        if solver_result is not None:
            bar_data = [
                (af.action, af.frequency, af.amount)
                for af in solver_result.strategy.actions
                if af.frequency >= 0.01
            ]
            self._bars.set_strategy(bar_data)
        else:
            self._bars.clear()

        # Analysis text
        lines = [f"Hand: {hero_cards_str}"]
        if decision.equity > 0:
            lines.append(f"Equity: {decision.equity:.1%}")
        if decision.pot_odds > 0:
            lines.append(f"Pot Odds: {decision.pot_odds:.1%}")
        if solver_result:
            if solver_result.ev != 0:
                lines.append(f"EV: {solver_result.ev:+.2f} bb")
            lines.append(f"Source: {solver_result.source} (confidence: {solver_result.confidence:.0%})")
        lines.append(f"\nReasoning: {decision.reasoning}")
        if opponent_advice:
            lines.append(f"\nOpponent: {opponent_advice}")

        self._analysis.setText("\n".join(lines))

    def show_gto_unavailable(self) -> None:
        """Display an amber warning that no GTO solution is available."""
        self._banner.setText("GTO SOLUTION UNAVAILABLE")
        self._banner.setStyleSheet(
            "QLabel { font-size: 18px; font-weight: bold; color: #92400e;"
            " background: #fef3c7; border-radius: 8px; }"
        )
        self._bars.clear()
        self._analysis.setText(
            "External solver not configured.\n\n"
            "Configure TexasSolver in ~/.poker_coach/solver_config.json:\n"
            '{\n'
            '  "solver_type": "texassolver",\n'
            '  "binary_path": "/path/to/console_solver",\n'
            '  "thread_count": 8,\n'
            '  "accuracy": 0.5\n'
            "}\n\n"
            "Without a configured solver, postflop GTO solutions\n"
            "cannot be computed. Preflop advice remains available."
        )

    def show_error(self, message: str) -> None:
        self._banner.setText("Error")
        self._banner.setStyleSheet(
            "QLabel { font-size: 18px; font-weight: bold; color: #fff;"
            " background: #ef4444; border-radius: 8px; }"
        )
        self._bars.clear()
        self._analysis.setText(message)

    def clear(self) -> None:
        self._banner.setText("Click SOLVE to get a recommendation")
        self._banner.setStyleSheet(
            "QLabel { font-size: 18px; font-weight: bold; color: #6b7280;"
            " background: #f3f4f6; border-radius: 8px; }"
        )
        self._bars.clear()
        self._analysis.clear()
