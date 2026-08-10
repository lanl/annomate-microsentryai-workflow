"""
AnomalyConstraintsSection — "Anomaly Constraints" section body for the
right activity bar's View Overlays tab.

Owns the area-threshold and proximity-threshold constraint controls that
used to live in the viewport's floating "⊿" popup menu, plus the two
public hooks (refresh_violations / update_units) window.py pushes
computed values into after each check run or calibration change.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QColor


class AnomalyConstraintsSection(QWidget):
    """Area/proximity anomaly-constraint controls, driven by an AnomalyConstraintModel."""

    def __init__(self, anomaly_constraint_model=None, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._model = None
        self._refreshing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._anomaly_enable_chk = QCheckBox("Enable")
        layout.addWidget(self._anomaly_enable_chk)

        # ── Area threshold section ──────────────────────────────────────
        div1 = QFrame()
        div1.setFrameShape(QFrame.HLine)
        div1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(div1)

        area_header = QLabel("Area Threshold")
        area_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(area_header)

        self._anomaly_area_chk = QCheckBox("Check Area")
        layout.addWidget(self._anomaly_area_chk)

        area_row = QHBoxLayout()
        area_row.setSpacing(4)
        area_row.addWidget(QLabel("Max Area:"))
        self._anomaly_area_spin = QDoubleSpinBox()
        self._anomaly_area_spin.setRange(0.0, 1e12)
        self._anomaly_area_spin.setDecimals(2)
        self._anomaly_area_spin.setSingleStep(1.0)
        self._anomaly_area_spin.setToolTip(
            "Annotations with area above this value will be highlighted"
        )
        area_row.addWidget(self._anomaly_area_spin, 1)
        self._anomaly_area_unit_lbl = QLabel("px²")
        self._anomaly_area_unit_lbl.setFixedWidth(30)
        area_row.addWidget(self._anomaly_area_unit_lbl)
        layout.addLayout(area_row)

        area_color_row = QHBoxLayout()
        area_color_row.setSpacing(4)
        area_color_row.addWidget(QLabel("Outline Color:"))
        self._anomaly_area_color_btn = QPushButton()
        self._anomaly_area_color_btn.setFixedSize(24, 20)
        self._anomaly_area_color_btn.setToolTip("Choose area violation outline color")
        self._apply_color_swatch(self._anomaly_area_color_btn, (255, 165, 0))
        area_color_row.addWidget(self._anomaly_area_color_btn)
        area_color_row.addStretch()
        layout.addLayout(area_color_row)

        self._anomaly_area_count_lbl = QLabel("")
        self._anomaly_area_count_lbl.setStyleSheet("color: #cc4400; font-weight: bold;")
        layout.addWidget(self._anomaly_area_count_lbl)

        # ── Distance threshold section ──────────────────────────────────
        div2 = QFrame()
        div2.setFrameShape(QFrame.HLine)
        div2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(div2)

        dist_header = QLabel("Proximity Threshold")
        dist_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(dist_header)

        self._anomaly_dist_chk = QCheckBox("Check Distance")
        layout.addWidget(self._anomaly_dist_chk)

        method_row = QHBoxLayout()
        method_row.setSpacing(4)
        method_row.addWidget(QLabel("Method:"))
        self._anomaly_centroid_radio = QRadioButton("Centroid")
        self._anomaly_edge_radio = QRadioButton("Edge")
        self._anomaly_centroid_radio.setChecked(True)
        self._anomaly_method_group = QButtonGroup(self)
        self._anomaly_method_group.addButton(self._anomaly_centroid_radio, 0)
        self._anomaly_method_group.addButton(self._anomaly_edge_radio, 1)
        method_row.addWidget(self._anomaly_centroid_radio)
        method_row.addWidget(self._anomaly_edge_radio)
        method_row.addStretch()
        layout.addLayout(method_row)

        dist_row = QHBoxLayout()
        dist_row.setSpacing(4)
        dist_row.addWidget(QLabel("Min Dist:"))
        self._anomaly_dist_spin = QDoubleSpinBox()
        self._anomaly_dist_spin.setRange(0.0, 1e12)
        self._anomaly_dist_spin.setDecimals(2)
        self._anomaly_dist_spin.setSingleStep(1.0)
        self._anomaly_dist_spin.setToolTip(
            "Annotation pairs closer than this distance will be highlighted"
        )
        dist_row.addWidget(self._anomaly_dist_spin, 1)
        self._anomaly_dist_unit_lbl = QLabel("px")
        self._anomaly_dist_unit_lbl.setFixedWidth(30)
        dist_row.addWidget(self._anomaly_dist_unit_lbl)
        layout.addLayout(dist_row)

        dist_color_row = QHBoxLayout()
        dist_color_row.setSpacing(4)
        dist_color_row.addWidget(QLabel("Line Color:"))
        self._anomaly_dist_color_btn = QPushButton()
        self._anomaly_dist_color_btn.setFixedSize(24, 20)
        self._anomaly_dist_color_btn.setToolTip("Choose proximity violation line color")
        self._apply_color_swatch(self._anomaly_dist_color_btn, (220, 50, 50))
        dist_color_row.addWidget(self._anomaly_dist_color_btn)
        dist_color_row.addStretch()
        layout.addLayout(dist_color_row)

        self._anomaly_dist_count_lbl = QLabel("")
        self._anomaly_dist_count_lbl.setStyleSheet("color: #cc4400; font-weight: bold;")
        layout.addWidget(self._anomaly_dist_count_lbl)

        # ── Wire signals ───────────────────────────────────────────────
        self._anomaly_enable_chk.toggled.connect(self._on_anomaly_enable_toggled)
        self._anomaly_area_chk.toggled.connect(self._on_anomaly_area_check_toggled)
        self._anomaly_area_spin.valueChanged.connect(self._on_anomaly_area_changed)
        self._anomaly_area_color_btn.clicked.connect(self._on_anomaly_area_color_clicked)
        self._anomaly_dist_chk.toggled.connect(self._on_anomaly_dist_check_toggled)
        self._anomaly_dist_spin.valueChanged.connect(self._on_anomaly_dist_changed)
        self._anomaly_centroid_radio.toggled.connect(self._on_anomaly_method_changed)
        self._anomaly_dist_color_btn.clicked.connect(self._on_anomaly_dist_color_clicked)

        if anomaly_constraint_model is not None:
            self.set_anomaly_constraint_model(anomaly_constraint_model)

    def set_anomaly_constraint_model(self, model) -> None:
        self._model = model
        model.constraints_changed.connect(self._refresh_controls)
        self._refresh_controls()

    def _on_anomaly_enable_toggled(self, checked: bool) -> None:
        if self._model is not None and not self._refreshing:
            self._model.set_enabled(checked)

    def _on_anomaly_area_check_toggled(self, checked: bool) -> None:
        if self._model is not None and not self._refreshing:
            self._model.set_area_check_enabled(checked)

    def _on_anomaly_area_changed(self, value: float) -> None:
        if self._model is not None and not self._refreshing:
            self._model.set_area_threshold(value)

    def _on_anomaly_dist_check_toggled(self, checked: bool) -> None:
        if self._model is not None and not self._refreshing:
            self._model.set_distance_check_enabled(checked)

    def _on_anomaly_dist_changed(self, value: float) -> None:
        if self._model is not None and not self._refreshing:
            self._model.set_distance_threshold(value)

    def _on_anomaly_method_changed(self, centroid_checked: bool) -> None:
        if self._model is not None and not self._refreshing:
            method = "centroid" if centroid_checked else "edge"
            self._model.set_distance_method(method)

    def _on_anomaly_area_color_clicked(self) -> None:
        if self._model is None:
            return
        r, g, b = self._model.area_color()
        color = QColorDialog.getColor(QColor(r, g, b), self, "Area Violation Color")
        if color.isValid():
            self._model.set_area_color((color.red(), color.green(), color.blue()))

    def _on_anomaly_dist_color_clicked(self) -> None:
        if self._model is None:
            return
        r, g, b = self._model.distance_color()
        color = QColorDialog.getColor(QColor(r, g, b), self, "Proximity Line Color")
        if color.isValid():
            self._model.set_distance_color((color.red(), color.green(), color.blue()))

    def _apply_color_swatch(self, btn, rgb: tuple) -> None:
        r, g, b = rgb
        btn.setStyleSheet(
            f"QPushButton {{ background-color: rgb({r},{g},{b}); "
            f"border: 1px solid #888; border-radius: 2px; }}"
        )

    def _refresh_controls(self) -> None:
        if self._model is None:
            return
        self._refreshing = True
        self._anomaly_enable_chk.setChecked(self._model.enabled())
        self._anomaly_area_chk.setChecked(self._model.area_check_enabled())
        self._anomaly_area_spin.setValue(self._model.area_threshold())
        self._apply_color_swatch(self._anomaly_area_color_btn, self._model.area_color())
        self._anomaly_dist_chk.setChecked(self._model.distance_check_enabled())
        self._anomaly_dist_spin.setValue(self._model.distance_threshold())
        self._anomaly_centroid_radio.setChecked(self._model.distance_method() == "centroid")
        self._anomaly_edge_radio.setChecked(self._model.distance_method() == "edge")
        self._apply_color_swatch(
            self._anomaly_dist_color_btn, self._model.distance_color()
        )
        self._refreshing = False

    # ------------------------------------------------------------------ #
    # External push hooks -- window.py calls these after running checks
    # or when the calibration unit changes.
    # ------------------------------------------------------------------ #

    def refresh_violations(self, area_count: int, dist_count: int) -> None:
        """Update the inline violation count labels with threshold-aware text."""
        if area_count > 0 and self._model is not None:
            t = self._model.area_threshold()
            unit = self._anomaly_area_unit_lbl.text()
            label = "defect" if area_count == 1 else "defects"
            self._anomaly_area_count_lbl.setText(f"{area_count} {label} > {t:g}{unit}")
        else:
            self._anomaly_area_count_lbl.setText("")

        if dist_count > 0 and self._model is not None:
            t = self._model.distance_threshold()
            unit = self._anomaly_dist_unit_lbl.text()
            label = "defect" if dist_count == 1 else "defects"
            self._anomaly_dist_count_lbl.setText(
                f"{dist_count} {label} within {t:g}{unit}"
            )
        else:
            self._anomaly_dist_count_lbl.setText("")

    def update_units(self, unit: str) -> None:
        """Update the unit labels (called when calibration changes)."""
        self._anomaly_area_unit_lbl.setText(f"{unit}²")
        self._anomaly_dist_unit_lbl.setText(unit)
