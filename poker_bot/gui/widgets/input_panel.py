"""Input panel widget for the poker coach GUI.

Contains hero card slots, board card slots, numeric inputs
(pot, bet, stack, opponents), street/game dropdowns, and
position radio buttons.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from poker_bot.gui.widgets.card_picker import CardPickerPopup, CardSlotButton


class InputPanel(QWidget):
    """Top input section: cards, pot/bet/stack, street, position, SOLVE button."""

    solve_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._hero_slots: list[CardSlotButton] = []
        self._board_slots: list[CardSlotButton] = []
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # --- Row 1: Cards ---
        cards_row = QHBoxLayout()

        # Hero cards
        hero_group = QGroupBox("Hero Cards")
        hero_layout = QHBoxLayout(hero_group)
        for _ in range(2):
            slot = CardSlotButton()
            slot.clicked.connect(lambda checked=False, s=slot: self._open_picker(s))
            self._hero_slots.append(slot)
            hero_layout.addWidget(slot)
        cards_row.addWidget(hero_group)

        # Board cards
        board_group = QGroupBox("Board")
        board_layout = QHBoxLayout(board_group)
        for _ in range(5):
            slot = CardSlotButton()
            slot.clicked.connect(lambda checked=False, s=slot: self._open_picker(s))
            self._board_slots.append(slot)
            board_layout.addWidget(slot)
        cards_row.addWidget(board_group)

        main_layout.addLayout(cards_row)

        # --- Row 2: Numeric inputs ---
        nums_row = QHBoxLayout()

        self._pot_spin = QDoubleSpinBox()
        self._pot_spin.setRange(0.0, 9999.0)
        self._pot_spin.setValue(3.0)
        self._pot_spin.setSuffix(" bb")
        self._pot_spin.setDecimals(1)
        nums_row.addWidget(QLabel("Pot:"))
        nums_row.addWidget(self._pot_spin)

        self._bet_spin = QDoubleSpinBox()
        self._bet_spin.setRange(0.0, 9999.0)
        self._bet_spin.setValue(0.0)
        self._bet_spin.setSuffix(" bb")
        self._bet_spin.setDecimals(1)
        nums_row.addWidget(QLabel("Bet:"))
        nums_row.addWidget(self._bet_spin)

        self._stack_spin = QDoubleSpinBox()
        self._stack_spin.setRange(0.5, 9999.0)
        self._stack_spin.setValue(50.0)
        self._stack_spin.setSuffix(" bb")
        self._stack_spin.setDecimals(1)
        nums_row.addWidget(QLabel("Stack:"))
        nums_row.addWidget(self._stack_spin)

        self._opp_spin = QSpinBox()
        self._opp_spin.setRange(1, 5)
        self._opp_spin.setValue(2)
        nums_row.addWidget(QLabel("Opp:"))
        nums_row.addWidget(self._opp_spin)

        main_layout.addLayout(nums_row)

        # --- Row 3: Dropdowns, position, solve ---
        controls_row = QHBoxLayout()

        self._street_combo = QComboBox()
        self._street_combo.addItems(["PREFLOP", "FLOP", "TURN", "RIVER"])
        controls_row.addWidget(QLabel("Street:"))
        controls_row.addWidget(self._street_combo)

        self._game_combo = QComboBox()
        self._game_combo.addItems(["Cash", "Tournament"])
        controls_row.addWidget(QLabel("Game:"))
        controls_row.addWidget(self._game_combo)

        self._position_combo = QComboBox()
        self._position_combo.addItems(["BTN", "CO", "MP", "UTG", "SB", "BB"])
        controls_row.addWidget(QLabel("Position:"))
        controls_row.addWidget(self._position_combo)

        # IP / OOP radio buttons
        self._ip_radio = QRadioButton("IP")
        self._oop_radio = QRadioButton("OOP")
        self._ip_radio.setChecked(True)
        pos_group = QButtonGroup(self)
        pos_group.addButton(self._ip_radio)
        pos_group.addButton(self._oop_radio)
        controls_row.addWidget(self._ip_radio)
        controls_row.addWidget(self._oop_radio)

        controls_row.addStretch()

        # SOLVE button
        self._solve_btn = QPushButton("SOLVE")
        self._solve_btn.setFixedHeight(36)
        self._solve_btn.setMinimumWidth(100)
        self._solve_btn.setStyleSheet(
            "QPushButton { background: #2563eb; color: white; font-weight: bold;"
            " font-size: 14px; border-radius: 6px; padding: 0 20px; }"
            "QPushButton:hover { background: #1d4ed8; }"
            "QPushButton:pressed { background: #1e40af; }"
        )
        self._solve_btn.clicked.connect(self.solve_requested.emit)
        controls_row.addWidget(self._solve_btn)

        main_layout.addLayout(controls_row)

    def _get_used_cards(self) -> set[str]:
        """Collect all currently selected cards."""
        used = set()
        for slot in self._hero_slots + self._board_slots:
            if slot.card:
                used.add(slot.card)
        return used

    def _open_picker(self, slot: CardSlotButton) -> None:
        """Open the card picker popup for a given slot."""
        used = self._get_used_cards()
        # Don't count the slot's own card as used
        if slot.card:
            used.discard(slot.card)
        popup = CardPickerPopup(used_cards=used, parent=self)
        popup.card_selected.connect(lambda c: setattr(slot, "card", c))
        popup.exec()

    # --- Public accessors for the presenter ---

    def get_hero_cards(self) -> list[str]:
        return [s.card for s in self._hero_slots if s.card]

    def get_board_cards(self) -> list[str]:
        return [s.card for s in self._board_slots if s.card]

    def get_pot_bb(self) -> float:
        return self._pot_spin.value()

    def get_bet_bb(self) -> float:
        return self._bet_spin.value()

    def get_stack_bb(self) -> float:
        return self._stack_spin.value()

    def get_num_opponents(self) -> int:
        return self._opp_spin.value()

    def get_street(self) -> str:
        return self._street_combo.currentText()

    def get_game_type(self) -> str:
        return self._game_combo.currentText()

    def get_position(self) -> str:
        return self._position_combo.currentText()

    def get_position_ip(self) -> bool:
        return self._ip_radio.isChecked()
