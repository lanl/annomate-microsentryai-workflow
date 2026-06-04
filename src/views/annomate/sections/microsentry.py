"""
MicrosentrySection — unified Microsentry controls panel for the AnnoMate right panel.

Layout (when model loaded):
  Load Model button
  Model name label
  [Heatmap] toggle  +  Transparency slider
  [Segmentation] toggle  +  Threshold slider
  [Accept AI Polygons] button
  ▸ Advanced Settings (collapsible)
      Simplify Tolerance slider
      Heatmap Minimum slider
"""

import os

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
)


def _slider_row(label_text: str, value_label: QLabel, slider: QSlider) -> QWidget:
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(1)
    top = QHBoxLayout()
    top.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label_text)
    lbl.setStyleSheet("font-size: 11px;")
    top.addWidget(lbl)
    top.addStretch()
    top.addWidget(value_label)
    v.addLayout(top)
    v.addWidget(slider)
    return w


class MicrosentrySection(QWidget):
    """Unified Microsentry controls: view toggles, sliders, and advanced settings.

    Signals:
        load_model_requested (): Load Model button clicked.
        settings_changed (): Any control changed (debounced 200 ms).
        accept_polygons_requested (): Accept AI Polygons button clicked.
    """

    load_model_requested = Signal()
    load_previous_model_requested = Signal()
    settings_changed = Signal()
    accept_polygons_requested = Signal()

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self.settings_changed)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Load buttons row (always visible)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(4)
        self._btn_load_prev = QPushButton("Load Previous")
        self._btn_load_prev.setToolTip("Reload the model saved with this project")
        self._btn_load_prev.clicked.connect(self.load_previous_model_requested)
        self._btn_load_new = QPushButton("Load New")
        self._btn_load_new.setToolTip("Browse for a new .pt model file")
        self._btn_load_new.clicked.connect(self.load_model_requested)
        btn_row.addWidget(self._btn_load_prev)
        btn_row.addWidget(self._btn_load_new)
        layout.addLayout(btn_row)

        # No-model label
        self._lbl_no_model = QLabel("No model loaded")
        self._lbl_no_model.setStyleSheet(
            "color: grey; font-style: italic; font-size: 11px;"
        )
        self._lbl_no_model.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._lbl_no_model)

        # ── Model-loaded body (hidden until set_model_loaded) ──────────── #
        self._loaded_widget = QWidget()
        lw = QVBoxLayout(self._loaded_widget)
        lw.setContentsMargins(0, 0, 0, 0)
        lw.setSpacing(2)

        model_info_row = QHBoxLayout()
        model_info_row.setContentsMargins(0, 0, 0, 0)
        model_info_row.setSpacing(6)
        self._lbl_model_file = QLabel("")
        self._lbl_model_file.setStyleSheet("font-size: 11px; font-weight: bold;")
        self._lbl_model_backend = QLabel("")
        self._lbl_model_backend.setStyleSheet("font-size: 10px; color: grey;")
        self._lbl_model_backend.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        model_info_row.addWidget(self._lbl_model_file)
        model_info_row.addWidget(self._lbl_model_backend, stretch=1)
        lw.addLayout(model_info_row)

        lw.addSpacing(4)

        # Heatmap toggle + transparency slider inline
        self._btn_heatmap = QToolButton()
        self._btn_heatmap.setText("Heatmap")
        self._btn_heatmap.setCheckable(True)
        self._btn_heatmap.setToolTip("Overlay anomaly heatmap on the canvas image")
        self._btn_heatmap.toggled.connect(self._debounce.start)
        self._alpha_val = QLabel("45%")
        self._alpha_val.setStyleSheet("font-size: 11px;")
        self._alpha_val.setFixedWidth(30)
        self._alpha = QSlider(Qt.Horizontal)
        self._alpha.setRange(0, 100)
        self._alpha.setValue(45)
        self._alpha.valueChanged.connect(
            lambda v: (self._alpha_val.setText(f"{v}%"), self._debounce.start())
        )
        heatmap_row = QHBoxLayout()
        heatmap_row.setContentsMargins(0, 0, 0, 0)
        heatmap_row.setSpacing(4)
        heatmap_row.addWidget(self._btn_heatmap)
        heatmap_row.addWidget(self._alpha, stretch=1)
        heatmap_row.addWidget(self._alpha_val)
        lw.addLayout(heatmap_row)

        # Segmentation toggle + threshold slider inline
        self._btn_seg = QToolButton()
        self._btn_seg.setText("Segmentation")
        self._btn_seg.setCheckable(True)
        self._btn_seg.setToolTip("Show AI segmentation polygons on the canvas")
        self._btn_seg.toggled.connect(self._on_seg_toggled)
        self._thresh_val = QLabel("95.0")
        self._thresh_val.setStyleSheet("font-size: 11px;")
        self._thresh_val.setFixedWidth(40)
        self._thresh = QSlider(Qt.Horizontal)
        self._thresh.setRange(0, 1000)
        self._thresh.setValue(950)
        self._thresh.valueChanged.connect(
            lambda v: (
                self._thresh_val.setText(f"{v / 10:.1f}"),
                self._debounce.start(),
            )
        )
        self._thresh_dec = QPushButton("<")
        self._thresh_dec.setFixedWidth(20)
        self._thresh_dec.clicked.connect(
            lambda: self._thresh.setValue(self._thresh.value() - 1)
        )
        self._thresh_inc = QPushButton(">")
        self._thresh_inc.setFixedWidth(20)
        self._thresh_inc.clicked.connect(
            lambda: self._thresh.setValue(self._thresh.value() + 1)
        )
        seg_row = QHBoxLayout()
        seg_row.setContentsMargins(0, 0, 0, 0)
        seg_row.setSpacing(4)
        seg_row.addWidget(self._btn_seg)
        seg_row.addWidget(self._thresh, stretch=1)
        seg_row.addWidget(self._thresh_dec)
        seg_row.addWidget(self._thresh_inc)
        seg_row.addWidget(self._thresh_val)
        lw.addLayout(seg_row)

        # Accept AI Polygons button
        self._btn_accept = QPushButton("Accept AI Polygons")
        self._btn_accept.setToolTip(
            "Add AI segmentation polygons as annotations on the active class"
        )
        self._btn_accept.setEnabled(False)
        self._btn_accept.clicked.connect(self.accept_polygons_requested)
        lw.addWidget(self._btn_accept)

        # ── Advanced Settings (inline collapsible) ──────────────────────── #
        self._btn_advanced = QToolButton()
        self._btn_advanced.setText("▸  Advanced Settings")
        self._btn_advanced.setCheckable(True)
        self._btn_advanced.setChecked(False)
        self._btn_advanced.setStyleSheet(
            "text-align: left; font-size: 11px; border: none;"
        )
        self._btn_advanced.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._btn_advanced.setSizePolicy(
            self._btn_advanced.sizePolicy().horizontalPolicy(),
            self._btn_advanced.sizePolicy().verticalPolicy(),
        )
        self._btn_advanced.toggled.connect(self._on_advanced_toggled)
        lw.addWidget(self._btn_advanced)

        self._advanced_widget = QWidget()
        aw = QVBoxLayout(self._advanced_widget)
        aw.setContentsMargins(8, 0, 0, 0)
        aw.setSpacing(4)

        self._epsilon_val = QLabel("12")
        self._epsilon_val.setStyleSheet("font-size: 11px;")
        self._epsilon_val.setFixedWidth(30)
        self._epsilon = QSlider(Qt.Horizontal)
        self._epsilon.setRange(0, 20)
        self._epsilon.setValue(12)
        self._epsilon.valueChanged.connect(
            lambda v: (self._epsilon_val.setText(str(v)), self._debounce.start())
        )
        aw.addWidget(
            _slider_row("Simplify Tolerance", self._epsilon_val, self._epsilon)
        )

        self._heat_min_val = QLabel("0%")
        self._heat_min_val.setStyleSheet("font-size: 11px;")
        self._heat_min_val.setFixedWidth(30)
        self._heat_min = QSlider(Qt.Horizontal)
        self._heat_min.setRange(0, 100)
        self._heat_min.setValue(0)
        self._heat_min.valueChanged.connect(
            lambda v: (self._heat_min_val.setText(f"{v}%"), self._debounce.start())
        )
        aw.addWidget(_slider_row("Heatmap Minimum", self._heat_min_val, self._heat_min))

        self._advanced_widget.setVisible(False)
        lw.addWidget(self._advanced_widget)

        layout.addWidget(self._loaded_widget)
        self._loaded_widget.setVisible(False)

    # ------------------------------------------------------------------ #
    # Internal slots
    # ------------------------------------------------------------------ #

    def _on_seg_toggled(self, checked: bool) -> None:
        self._btn_accept.setEnabled(checked)
        self._debounce.start()

    def _on_advanced_toggled(self, checked: bool) -> None:
        self._advanced_widget.setVisible(checked)
        self._btn_advanced.setText(f"{'▾' if checked else '▸'}  Advanced Settings")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_model_loaded(self, name: str, path: str = "") -> None:
        filename = os.path.basename(path) if path else name
        self._lbl_model_file.setText(filename)
        self._lbl_model_backend.setText(name)
        self._lbl_no_model.setVisible(False)
        self._loaded_widget.setVisible(True)

    def set_no_model(self) -> None:
        self._lbl_model_file.setText("")
        self._lbl_model_backend.setText("")
        self._lbl_no_model.setVisible(True)
        self._loaded_widget.setVisible(False)

    def get_settings(self) -> dict:
        return {
            "heatmap_enabled": self._btn_heatmap.isChecked(),
            "seg_enabled": self._btn_seg.isChecked(),
            "seg_pct": self._thresh.value() / 10.0,
            "alpha": self._alpha.value() / 100.0,
            "epsilon": self._epsilon.value(),
            "heat_min": self._heat_min.value(),
        }
