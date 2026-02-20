"""QSS stylesheet constants for the poker coach GUI."""

APP_STYLESHEET = """
QMainWindow {
    background: #f8fafc;
}

QGroupBox {
    font-weight: bold;
    font-size: 12px;
    color: #1f2937;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 14px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #4b5563;
}

QLabel {
    font-size: 12px;
    color: #1f2937;
}

QDoubleSpinBox, QSpinBox {
    padding: 4px 6px;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    background: white;
    min-width: 70px;
}

QDoubleSpinBox:focus, QSpinBox:focus {
    border-color: #2563eb;
}

QComboBox {
    padding: 4px 8px;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    background: white;
    min-width: 90px;
}

QComboBox:focus {
    border-color: #2563eb;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QLineEdit {
    padding: 4px 8px;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    background: white;
}

QLineEdit:focus {
    border-color: #2563eb;
}

QRadioButton {
    font-size: 12px;
    spacing: 4px;
}

QTextEdit {
    border: 1px solid #e5e7eb;
    border-radius: 4px;
}

QScrollArea {
    border: none;
}
"""
