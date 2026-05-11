"""
MicrosentrySection — unified Microsentry controls panel for the AnnoMate right panel.

Layout (when model loaded):
  Load Model button
  Model name label
  [Heatmap] toggle  +  Overlay opacity slider
  [Segmentation] toggle  +  Detection sensitivity slider
  [Accept AI Polygons] button
  ▸ Advanced Settings (collapsible) + Help button
      Mask smoothness slider
      Boundary simplification slider
      Heatmap floor slider
"""

import os

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QDialog,
    QLabel,
    QPushButton,
    QSlider,
    QTextBrowser,
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

        # Heatmap toggle + overlay opacity slider inline
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
        self._alpha.setToolTip(
            "Controls how visible the heatmap overlay is on top of the image."
        )
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

        # Segmentation toggle + detection sensitivity slider inline
        self._btn_seg = QToolButton()
        self._btn_seg.setText("Segmentation")
        self._btn_seg.setCheckable(True)
        self._btn_seg.setToolTip("Show AI segmentation polygons on the canvas")
        self._btn_seg.toggled.connect(self._on_seg_toggled)
        self._thresh_val = QLabel("95")
        self._thresh_val.setStyleSheet("font-size: 11px;")
        self._thresh_val.setFixedWidth(30)
        self._thresh = QSlider(Qt.Horizontal)
        self._thresh.setRange(0, 100)
        self._thresh.setValue(95)
        self._thresh.setToolTip(
            "Controls the percentile cutoff used to generate anomaly masks; "
            "higher values keep only stronger anomaly regions."
        )
        self._thresh.valueChanged.connect(
            lambda v: (self._thresh_val.setText(str(v)), self._debounce.start())
        )
        seg_row = QHBoxLayout()
        seg_row.setContentsMargins(0, 0, 0, 0)
        seg_row.setSpacing(4)
        seg_row.addWidget(self._btn_seg)
        seg_row.addWidget(self._thresh, stretch=1)
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

        self._btn_advanced_help = QPushButton("Help")
        self._btn_advanced_help.setToolTip(
            "Explain what the advanced MicroSentryAI settings do"
        )
        self._btn_advanced_help.clicked.connect(self._show_advanced_help)

        advanced_header = QHBoxLayout()
        advanced_header.setContentsMargins(0, 0, 0, 0)
        advanced_header.setSpacing(4)
        advanced_header.addWidget(self._btn_advanced, stretch=1)
        advanced_header.addWidget(self._btn_advanced_help)
        lw.addLayout(advanced_header)

        self._advanced_widget = QWidget()
        aw = QVBoxLayout(self._advanced_widget)
        aw.setContentsMargins(8, 0, 0, 0)
        aw.setSpacing(4)

        self._sigma_val = QLabel("4")
        self._sigma_val.setStyleSheet("font-size: 11px;")
        self._sigma_val.setFixedWidth(30)
        self._sigma = QSlider(Qt.Horizontal)
        self._sigma.setRange(0, 16)
        self._sigma.setValue(4)
        self._sigma.setToolTip(
            "Controls how much Gaussian smoothing is applied before mask generation."
        )
        self._sigma.valueChanged.connect(
            lambda v: (self._sigma_val.setText(str(v)), self._debounce.start())
        )
        aw.addWidget(_slider_row("Mask smoothness", self._sigma_val, self._sigma))

        self._epsilon_val = QLabel("12")
        self._epsilon_val.setStyleSheet("font-size: 11px;")
        self._epsilon_val.setFixedWidth(30)
        self._epsilon = QSlider(Qt.Horizontal)
        self._epsilon.setRange(0, 20)
        self._epsilon.setValue(12)
        self._epsilon.setToolTip(
            "Controls polygon simplification; higher values create simpler boundaries."
        )
        self._epsilon.valueChanged.connect(
            lambda v: (self._epsilon_val.setText(str(v)), self._debounce.start())
        )
        aw.addWidget(
            _slider_row("Boundary simplification", self._epsilon_val, self._epsilon)
        )

        self._heat_min_val = QLabel("0%")
        self._heat_min_val.setStyleSheet("font-size: 11px;")
        self._heat_min_val.setFixedWidth(30)
        self._heat_min = QSlider(Qt.Horizontal)
        self._heat_min.setRange(0, 100)
        self._heat_min.setValue(0)
        self._heat_min.setToolTip(
            "Hides lower-intensity heatmap values below this percentile."
        )
        self._heat_min.valueChanged.connect(
            lambda v: (self._heat_min_val.setText(f"{v}%"), self._debounce.start())
        )
        aw.addWidget(_slider_row("Heatmap floor", self._heat_min_val, self._heat_min))

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

    def _show_advanced_help(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("MicroSentryAI Advanced Settings")
        dialog.resize(560, 420)

        layout = QVBoxLayout(dialog)

        help_view = QTextBrowser()
        help_view.setOpenExternalLinks(False)
        help_view.setHtml(
            """
            <h2>Advanced Settings Help</h2>
            <p>
              These controls tune how MicroSentryAI turns model heatmaps into
              visual overlays and editable segmentation polygons.
            </p>

            <h3>Mask smoothness</h3>
            <p>
              Controls Gaussian smoothing before mask generation. Increase it
              when the heatmap is noisy or produces speckled polygons. Lower it
              when small defects are being blurred or missed.
            </p>
            <p><b>Tradeoff:</b> higher values reduce noise but can merge nearby
              regions or soften small anomalies.</p>

            <h3>Boundary simplification</h3>
            <p>
              Controls how aggressively AI-generated polygon boundaries are
              simplified after segmentation. Increase it when polygons have too
              many jagged points. Lower it when the boundary shape needs to stay
              closer to the original mask.
            </p>
            <p><b>Tradeoff:</b> higher values create cleaner, simpler polygons
              but may lose fine shape detail.</p>

            <h3>Heatmap floor</h3>
            <p>
              Hides lower-intensity heatmap values below the selected percentile
              so weak background signal does not dominate the overlay. Increase
              it when the heatmap looks cluttered. Lower it when you want to see
              weaker model responses.
            </p>
            <p><b>Tradeoff:</b> higher values make the overlay cleaner but can
              hide subtle anomalies.</p>
            """
        )
        layout.addWidget(help_view)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        dialog.exec()

    def _refresh_value_labels(self) -> None:
        self._alpha_val.setText(f"{self._alpha.value()}%")
        self._thresh_val.setText(str(self._thresh.value()))
        self._sigma_val.setText(str(self._sigma.value()))
        self._epsilon_val.setText(str(self._epsilon.value()))
        self._heat_min_val.setText(f"{self._heat_min.value()}%")

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

    def set_settings(self, settings: dict) -> None:
        """Apply persisted MicroSentryAI user settings to the controls."""
        if not settings:
            return

        widgets = (
            self._btn_heatmap,
            self._btn_seg,
            self._alpha,
            self._thresh,
            self._sigma,
            self._epsilon,
            self._heat_min,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self._btn_heatmap.setChecked(
                bool(settings.get("heatmap_enabled", self._btn_heatmap.isChecked()))
            )
            self._btn_seg.setChecked(
                bool(settings.get("seg_enabled", self._btn_seg.isChecked()))
            )
            self._alpha.setValue(int(float(settings.get("alpha", 0.45)) * 100))
            self._thresh.setValue(int(settings.get("seg_pct", 95)))
            self._sigma.setValue(int(settings.get("sigma", 4)))
            self._epsilon.setValue(int(settings.get("epsilon", 12)))
            self._heat_min.setValue(int(settings.get("heat_min", 0)))
        finally:
            for widget in widgets:
                widget.blockSignals(False)

        self._btn_accept.setEnabled(self._btn_seg.isChecked())
        self._refresh_value_labels()

    def get_settings(self) -> dict:
        return {
            "heatmap_enabled": self._btn_heatmap.isChecked(),
            "seg_enabled": self._btn_seg.isChecked(),
            "seg_pct": self._thresh.value(),
            "alpha": self._alpha.value() / 100.0,
            "sigma": self._sigma.value(),
            "epsilon": self._epsilon.value(),
            "heat_min": self._heat_min.value(),
        }
