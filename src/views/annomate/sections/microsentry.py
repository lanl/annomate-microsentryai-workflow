"""
MicrosentrySection — unified Microsentry controls panel for the AnnoMate right panel.

Layout (when model loaded):
  Load New Model button
  Model name line + backend line (only shown when a model is actually loaded
      for inference — not shown for cached-only viewing)
  Cached Model dropdown (only shown when >1 model has cached results)
  Unsaved-scores indicator (only shown when the active model is dirty)
  [Enable Heatmap] toggle
      Transparency slider (+ nudge buttons)
  [Enable Segmentation] toggle
      Threshold slider (+ nudge buttons)
  [Accept AI Polygons] button
  chevron_right  Advanced Settings (collapsible)
      Simplify Tolerance slider
      Heatmap Minimum slider
      Heatmap Ceiling slider
      Heatmap Gamma slider
      Heatmap Colormap dropdown
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
    QComboBox,
)

from views.icons import material_icon

from ._shared import _toggle_button

_ICON_ADVANCED_EXPANDED = "expand_more"
_ICON_ADVANCED_COLLAPSED = "chevron_right"

# (display label, colormap key understood by ImageLabel.set_heatmap_layer)
HEATMAP_COLORMAPS = [
    ("Inferno", "inferno"),
    ("Magma", "magma"),
    ("Viridis", "viridis"),
    ("Turbo", "turbo"),
    ("Jet", "jet"),
    ("Hot", "hot"),
]


def _slider_row(
    label_text: str, value_label: QLabel, slider: QSlider, trailing=None
) -> QWidget:
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
    bottom = QHBoxLayout()
    bottom.setContentsMargins(0, 0, 0, 0)
    bottom.setSpacing(4)
    bottom.addWidget(slider, stretch=1)
    for widget in trailing or []:
        bottom.addWidget(widget)
    v.addLayout(bottom)
    return w


class MicrosentrySection(QWidget):
    """Unified Microsentry controls: view toggles, sliders, and advanced settings.

    Signals:
        load_model_requested (): Load Model button clicked.
        settings_changed (): Any control changed (debounced 200 ms).
        accept_polygons_requested (): Accept AI Polygons button clicked.
    """

    load_model_requested = Signal()
    settings_changed = Signal()
    accept_polygons_requested = Signal()
    cached_model_changed = Signal(str)  # model key the user picked from the dropdown

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self.settings_changed)
        self._updating_cached_model = False
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Load button (always visible)
        self._btn_load_new = QPushButton("Load New Model")
        self._btn_load_new.setToolTip("Browse for a new .pt model file")
        self._btn_load_new.setEnabled(False)
        self._btn_load_new.setObjectName("microsentryLoadNewButton")
        self._btn_load_new.clicked.connect(self.load_model_requested)
        layout.addWidget(self._btn_load_new)

        # Save-project hint (shown when no project is saved)
        self._lbl_save_hint = QLabel("Save the project first to enable model loading.")
        self._lbl_save_hint.setStyleSheet("color: #b05000; font-size: 11px;")
        self._lbl_save_hint.setWordWrap(True)
        layout.addWidget(self._lbl_save_hint)

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

        # Model name + backend, on their own lines — only shown while an
        # actual PyTorch model is loaded and available for inference, never
        # for cached-only viewing (see set_model_loaded/set_scoremaps_loaded).
        self._model_info_widget = QWidget()
        model_info_col = QVBoxLayout(self._model_info_widget)
        model_info_col.setContentsMargins(0, 0, 0, 0)
        model_info_col.setSpacing(0)
        self._lbl_model_file = QLabel("")
        self._lbl_model_file.setStyleSheet("font-size: 11px; font-weight: bold;")
        self._lbl_model_backend = QLabel("")
        self._lbl_model_backend.setStyleSheet("font-size: 10px; color: grey;")
        model_info_col.addWidget(self._lbl_model_file)
        model_info_col.addWidget(self._lbl_model_backend)
        self._model_info_widget.setVisible(False)
        lw.addWidget(self._model_info_widget)

        # Cached-model selector — shown only when this project has cached
        # results for more than one model. Switching here just swaps which
        # model's cached scores/heatmap are displayed; it does not load
        # weights or run new inference (see "Load New Model" for that).
        self._cached_model_row_widget = QWidget()
        cached_model_row = QHBoxLayout(self._cached_model_row_widget)
        cached_model_row.setContentsMargins(0, 0, 0, 0)
        cached_model_lbl = QLabel("Saved Results")
        cached_model_lbl.setStyleSheet("font-size: 11px;")
        self._cached_model = QComboBox()
        # Mouse-only: QComboBox's default keyboard focus enables type-ahead
        # (e.g. pressing "a" jumps to an item starting with "a"), which was
        # swallowing the app's A/D image-navigation shortcuts whenever this
        # combo box still had focus after a selection.
        self._cached_model.setFocusPolicy(Qt.NoFocus)
        self._cached_model.currentIndexChanged.connect(self._on_cached_model_changed)
        cached_model_row.addWidget(cached_model_lbl)
        cached_model_row.addStretch()
        cached_model_row.addWidget(self._cached_model)
        self._cached_model_row_widget.setVisible(False)
        lw.addWidget(self._cached_model_row_widget)

        # Unsaved-scores indicator — visible whenever the active model's
        # heatmaps have changes that haven't been written to disk yet.
        self._lbl_unsaved_scores = QLabel("Unsaved scores - Save Project to keep them")
        self._lbl_unsaved_scores.setStyleSheet("color: #b05000; font-size: 11px;")
        self._lbl_unsaved_scores.setWordWrap(True)
        self._lbl_unsaved_scores.setVisible(False)
        lw.addWidget(self._lbl_unsaved_scores)

        lw.addSpacing(4)

        # Heatmap toggle, transparency slider underneath
        self._btn_heatmap = _toggle_button("Enable Heatmap")
        self._btn_heatmap.setObjectName("microsentryHeatmapButton")
        self._btn_heatmap.setToolTip("Overlay anomaly heatmap on the canvas image")
        self._btn_heatmap.toggled.connect(self._debounce.start)
        lw.addWidget(self._btn_heatmap)

        self._alpha_val = QLabel("45%")
        self._alpha_val.setStyleSheet("font-size: 11px;")
        self._alpha_val.setFixedWidth(30)
        self._alpha = QSlider(Qt.Horizontal)
        self._alpha.setRange(0, 100)
        self._alpha.setValue(45)
        self._alpha.valueChanged.connect(
            lambda v: (self._alpha_val.setText(f"{v}%"), self._debounce.start())
        )
        self._alpha_dec = QPushButton(material_icon("chevron_left", size=12), "")
        self._alpha_dec.setFixedWidth(20)
        self._alpha_dec.clicked.connect(
            lambda: self._alpha.setValue(self._alpha.value() - 1)
        )
        self._alpha_inc = QPushButton(material_icon("chevron_right", size=12), "")
        self._alpha_inc.setFixedWidth(20)
        self._alpha_inc.clicked.connect(
            lambda: self._alpha.setValue(self._alpha.value() + 1)
        )
        lw.addWidget(
            _slider_row(
                "Transparency",
                self._alpha_val,
                self._alpha,
                [self._alpha_dec, self._alpha_inc],
            )
        )

        # Segmentation toggle, threshold slider underneath
        self._btn_seg = _toggle_button("Enable Segmentation")
        self._btn_seg.setToolTip("Show AI segmentation polygons on the canvas")
        self._btn_seg.toggled.connect(self._on_seg_toggled)
        lw.addWidget(self._btn_seg)

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
        self._thresh_dec = QPushButton(material_icon("chevron_left", size=12), "")
        self._thresh_dec.setFixedWidth(20)
        self._thresh_dec.clicked.connect(
            lambda: self._thresh.setValue(self._thresh.value() - 1)
        )
        self._thresh_inc = QPushButton(material_icon("chevron_right", size=12), "")
        self._thresh_inc.setFixedWidth(20)
        self._thresh_inc.clicked.connect(
            lambda: self._thresh.setValue(self._thresh.value() + 1)
        )
        lw.addWidget(
            _slider_row(
                "Threshold",
                self._thresh_val,
                self._thresh,
                [self._thresh_dec, self._thresh_inc],
            )
        )

        # Accept AI Polygons button
        self._btn_accept = QPushButton(material_icon("check"), "Accept AI Polygons")
        self._btn_accept.setObjectName("microsentryAcceptButton")
        self._btn_accept.setToolTip(
            "Add AI segmentation polygons as annotations on the active class"
        )
        self._btn_accept.setEnabled(False)
        self._btn_accept.clicked.connect(self.accept_polygons_requested)
        lw.addWidget(self._btn_accept)

        # ── Advanced Settings (inline collapsible) ──────────────────────── #
        self._btn_advanced = QToolButton()
        self._btn_advanced.setIcon(material_icon(_ICON_ADVANCED_COLLAPSED, size=14))
        self._btn_advanced.setText("  Advanced Settings")
        self._btn_advanced.setCheckable(True)
        self._btn_advanced.setChecked(False)
        self._btn_advanced.setStyleSheet(
            "text-align: left; font-size: 11px; border: none;"
        )
        self._btn_advanced.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
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

        self._heat_min_val = QLabel("48%")
        self._heat_min_val.setStyleSheet("font-size: 11px;")
        self._heat_min_val.setFixedWidth(30)
        self._heat_min = QSlider(Qt.Horizontal)
        self._heat_min.setRange(0, 100)
        self._heat_min.setValue(48)
        self._heat_min.valueChanged.connect(
            lambda v: (self._heat_min_val.setText(f"{v}%"), self._debounce.start())
        )
        aw.addWidget(_slider_row("Heatmap Minimum", self._heat_min_val, self._heat_min))

        self._heat_ceiling_val = QLabel("62%")
        self._heat_ceiling_val.setStyleSheet("font-size: 11px;")
        self._heat_ceiling_val.setFixedWidth(30)
        self._heat_ceiling = QSlider(Qt.Horizontal)
        self._heat_ceiling.setRange(50, 100)
        self._heat_ceiling.setValue(62)
        self._heat_ceiling.valueChanged.connect(
            lambda v: (self._heat_ceiling_val.setText(f"{v}%"), self._debounce.start())
        )
        aw.addWidget(
            _slider_row("Heatmap Ceiling", self._heat_ceiling_val, self._heat_ceiling)
        )

        self._heat_gamma_val = QLabel("0.60")
        self._heat_gamma_val.setStyleSheet("font-size: 11px;")
        self._heat_gamma_val.setFixedWidth(30)
        self._heat_gamma = QSlider(Qt.Horizontal)
        self._heat_gamma.setRange(10, 100)
        self._heat_gamma.setValue(60)
        self._heat_gamma.valueChanged.connect(
            lambda v: (
                self._heat_gamma_val.setText(f"{v / 100:.2f}"),
                self._debounce.start(),
            )
        )
        aw.addWidget(_slider_row("Heatmap Gamma", self._heat_gamma_val, self._heat_gamma))

        colormap_row = QHBoxLayout()
        colormap_row.setContentsMargins(0, 0, 0, 0)
        colormap_lbl = QLabel("Heatmap Colormap")
        colormap_lbl.setStyleSheet("font-size: 11px;")
        self._colormap = QComboBox()
        for display_label, key in HEATMAP_COLORMAPS:
            self._colormap.addItem(display_label, key)
        self._colormap.currentIndexChanged.connect(self._debounce.start)
        colormap_row.addWidget(colormap_lbl)
        colormap_row.addStretch()
        colormap_row.addWidget(self._colormap)
        aw.addLayout(colormap_row)

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
        self._btn_advanced.setIcon(
            material_icon(
                _ICON_ADVANCED_EXPANDED if checked else _ICON_ADVANCED_COLLAPSED,
                size=14,
            )
        )

    def _on_cached_model_changed(self, index: int) -> None:
        if self._updating_cached_model or index < 0:
            return
        key = self._cached_model.itemData(index)
        if key:
            self.cached_model_changed.emit(key)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_project_saved(self, has_project: bool) -> None:
        self._btn_load_new.setEnabled(has_project)
        self._lbl_save_hint.setVisible(not has_project)

    def set_model_loaded(self, name: str, path: str = "") -> None:
        """A real PyTorch model is loaded and available for inference."""
        filename = os.path.basename(path) if path else name
        self._lbl_model_file.setText(filename)
        self._lbl_model_backend.setText(name)
        self._model_info_widget.setVisible(True)
        self._lbl_no_model.setVisible(False)
        self._loaded_widget.setVisible(True)

    def set_scoremaps_loaded(self) -> None:
        """Cached results exist for this project, but no model is loaded.

        The rest of the panel (cached-model dropdown, heatmap controls)
        still works against the cached data, but the model name/backend
        lines only apply to an actual loaded model, so stay hidden here.
        """
        self._model_info_widget.setVisible(False)
        self._lbl_no_model.setVisible(False)
        self._loaded_widget.setVisible(True)

    def set_no_model(self) -> None:
        self._model_info_widget.setVisible(False)
        self._lbl_model_file.setText("")
        self._lbl_model_backend.setText("")
        self._lbl_no_model.setVisible(True)
        self._loaded_widget.setVisible(False)

    def set_known_models(self, models: dict, active_key: str) -> None:
        """Populate the cached-model dropdown, shown only when >1 model is known.

        Args:
            models (dict): ``{key: {"model_path": str, ...}}`` registry.
            active_key (str): Currently active model's key, pre-selected.
        """
        self._updating_cached_model = True
        try:
            self._cached_model.clear()
            for key in models:
                self._cached_model.addItem(key, key)
            idx = self._cached_model.findData(active_key)
            if idx >= 0:
                self._cached_model.setCurrentIndex(idx)
        finally:
            self._updating_cached_model = False
        self._cached_model_row_widget.setVisible(len(models) > 1)

    def set_scores_dirty(self, dirty: bool) -> None:
        """Show or hide the unsaved-scores warning indicator."""
        self._lbl_unsaved_scores.setVisible(dirty)

    def get_settings(self) -> dict:
        return {
            "heatmap_enabled": self._btn_heatmap.isChecked(),
            "seg_enabled": self._btn_seg.isChecked(),
            "seg_pct": self._thresh.value() / 10.0,
            "alpha": self._alpha.value() / 100.0,
            "epsilon": self._epsilon.value(),
            "heat_min": self._heat_min.value(),
            "heat_ceiling": self._heat_ceiling.value(),
            "heat_gamma": self._heat_gamma.value() / 100.0,
            "colormap": self._colormap.currentData(),
        }
