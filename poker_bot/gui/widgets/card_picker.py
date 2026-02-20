"""Card picker popup dialog.

Displays a 13x4 grid of rank x suit buttons. Clicking a card
selects it and closes the popup. Already-selected cards elsewhere
in the UI are disabled to prevent duplicates.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
_SUITS = ["s", "h", "d", "c"]

_SUIT_SYMBOLS = {"s": "\u2660", "h": "\u2665", "d": "\u2666", "c": "\u2663"}
_SUIT_COLORS = {
    "s": "#1a1a2e",
    "h": "#e63946",
    "d": "#2a7de1",
    "c": "#2d6a4f",
}


class CardPickerPopup(QDialog):
    """A 13x4 grid dialog for selecting a single card.

    Emits card_selected(str) with the 2-char card string (e.g. 'Ah').
    """

    card_selected = Signal(str)

    def __init__(self, used_cards: set[str] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pick a Card")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._used = used_cards or set()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setSpacing(2)

        for col, suit in enumerate(_SUITS):
            for row, rank in enumerate(_RANKS):
                card_str = f"{rank}{suit}"
                symbol = _SUIT_SYMBOLS[suit]
                btn = QPushButton(f"{rank}{symbol}")
                btn.setFixedSize(48, 36)
                btn.setStyleSheet(
                    f"QPushButton {{"
                    f"  color: {_SUIT_COLORS[suit]}; font-weight: bold; font-size: 13px;"
                    f"  border: 1px solid #ccc; border-radius: 4px; background: #fafafa;"
                    f"}}"
                    f"QPushButton:hover {{ background: #e0e7ff; }}"
                    f"QPushButton:disabled {{ color: #ccc; background: #f0f0f0; }}"
                )
                if card_str in self._used:
                    btn.setEnabled(False)
                else:
                    btn.clicked.connect(lambda checked=False, c=card_str: self._pick(c))
                layout.addWidget(btn, row, col)

    def _pick(self, card_str: str) -> None:
        self.card_selected.emit(card_str)
        self.accept()


class CardSlotButton(QPushButton):
    """A button representing a single card slot (hero hand or board).

    Displays '?' when empty, or the colored card when set.
    Click opens CardPickerPopup. Right-click clears the card.
    """

    card_changed = Signal()  # Emitted when card is set or cleared

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("?", parent)
        self._card: str = ""
        self.setFixedSize(56, 40)
        self.setCursor(Qt.PointingHandCursor)
        self._update_display()

    @property
    def card(self) -> str:
        return self._card

    @card.setter
    def card(self, value: str) -> None:
        self._card = value
        self._update_display()
        self.card_changed.emit()

    def clear_card(self) -> None:
        self.card = ""

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self.clear_card()
        else:
            super().mousePressEvent(event)

    def _update_display(self) -> None:
        if not self._card:
            self.setText("?")
            self.setStyleSheet(
                "QPushButton { font-size: 16px; font-weight: bold; color: #999;"
                " border: 2px dashed #ccc; border-radius: 6px; background: #fafafa; }"
                "QPushButton:hover { border-color: #888; background: #f0f0ff; }"
            )
        else:
            rank = self._card[0]
            suit = self._card[1]
            symbol = _SUIT_SYMBOLS.get(suit, suit)
            color = _SUIT_COLORS.get(suit, "#000")
            self.setText(f"{rank}{symbol}")
            self.setStyleSheet(
                f"QPushButton {{ font-size: 14px; font-weight: bold; color: {color};"
                f" border: 2px solid {color}; border-radius: 6px; background: #fff; }}"
                f"QPushButton:hover {{ background: #e0e7ff; }}"
            )
