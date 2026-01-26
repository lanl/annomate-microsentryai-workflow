"""
Application Styling Module.

This module contains the centralized Qt Style Sheets (QSS) for the AnnoMate
application. It defines the visual appearance of widgets, splitters, and
controls to ensure a consistent theme across the UI.
"""

# -----------------------------------------------------------------------------
# STYLESHEETS
# -----------------------------------------------------------------------------

MAIN_STYLESHEET = """
    /* --- General Widget Styling --- */
    QWidget {
        font-size: 13px;
        color: #333333;
    }

    /* --- Buttons --- */
    QPushButton {
        background-color: #FFFFFF;
        border: 1px solid #C4C4C4;
        border-radius: 5px;
        padding: 5px 5px;
        min-height: 5px;
    }
    QPushButton:hover {
        background-color: #C7E0F2;
        border-color: #A0A0A0;
    }
    QPushButton:pressed {
        background-color: #A5C0D4;
        border-color: #707070;
    }
    QPushButton:checked {
        background-color: #95B6CF;
        border: 1px solid #707070;
    }

    /* --- Dropdowns (ComboBox) --- */
    QComboBox {
        background-color: #FFFFFF;
        border: 1px solid #C4C4C4;
        border-radius: 5px;
        padding: 4px;
        padding-right: 30px; /* Space for the indicator */
    }
    QComboBox:hover {
        background-color: #F2F2F2;
        border-color: #A0A0A0;
    }

    /* The click area for the dropdown */
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 30px;
        border-left-width: 1px;
        border-left-color: #C4C4C4;
        border-left-style: solid;
        border-top-right-radius: 5px;
        border-bottom-right-radius: 5px;
        background: #FAFAFA;
    }
    QComboBox::drop-down:hover {
        background: #E0E0E0;
    }

    /* THE FIX: A Geometric Dot Indicator
       Since transparency is buggy on your system (black boxes) and images are 
       failing to load, we use a solid 6px circle. This is bulletproof. */
    QComboBox::down-arrow {
        image: none;
        width: 6px;
        height: 6px;
        background: #555555; /* Dark Grey Dot */
        border-radius: 3px;  /* Makes it a circle */
        margin-right: 2px;
    }

    /* Popup List Styling */
    QComboBox QAbstractItemView {
        border: 1px solid #C4C4C4;
        background-color: #FFFFFF;
        outline: none;
        padding: 4px;
    }
    QComboBox QAbstractItemView::item {
        padding: 4px;
        min-height: 20px;
        color: #333333;
    }
    QComboBox QAbstractItemView::item:selected {
        background-color: #E8F0FE;
        color: #000000;
        border-radius: 3px;
    }

    /* --- Lists & Tables --- */
    QListWidget, QTableWidget {
        border: 1px solid #C4C4C4;
        background-color: #FFFFFF;
        alternate-background-color: #FAFAFA;
        outline: none;
    }
    QListWidget::item:hover, QTableWidget::item:hover {
        background-color: #E8F0FE;
    }
    QListWidget::item:selected, QTableWidget::item:selected {
        background-color: #B3D7FF;
        color: #000000;
    }

    /* --- Text Inputs --- */
    QLineEdit, QTextEdit {
        border: 1px solid #C4C4C4;
        border-radius: 4px;
        padding: 4px;
        background-color: #FFFFFF;
    }
    QLineEdit:focus, QTextEdit:focus {
        border: 1px solid #4A90E2;
    }
    
    /* --- SpinBoxes (Number Inputs) --- */
    QSpinBox, QDoubleSpinBox {
        background-color: #FFFFFF;
        border: 1px solid #C4C4C4;
        border-radius: 5px;
        padding: 4px;
        padding-right: 15px;
        min-height: 20px;
    }
    QSpinBox:hover, QDoubleSpinBox:hover {
        background-color: #F2F2F2;
        border-color: #A0A0A0;
    }
    QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid #4A90E2;
    }

    /* SpinBox Buttons */
    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 16px;
        border-left: 1px solid #C4C4C4;
        border-bottom: 1px solid #C4C4C4;
        border-top-right-radius: 5px;
        background: #FAFAFA;
    }
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 16px;
        border-left: 1px solid #C4C4C4;
        border-bottom-right-radius: 5px;
        background: #FAFAFA;
    }
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
        background-color: #E0E0E0;
    }
    
    /* SpinBox Indicators (Dots) */
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
        image: none;
        width: 4px;
        height: 4px;
        background: #555555;
        border-radius: 2px;
    }
"""

SPLITTER_STYLE = """
    /* --- Splitter Handle Styling --- */
    QSplitter::handle {
        background-color: #004253;
        border: 1px solid #001021;
        border-radius: 2px;
        image: none;
    }
    QSplitter::handle:hover {
        background-color: #F17221;
        border: 1px solid #BE4003;
    }
    QSplitter::handle:pressed {
        background-color: #FFAD4D;
    }
"""