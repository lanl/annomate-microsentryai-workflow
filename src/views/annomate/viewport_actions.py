from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)


class _UpwardMenuToolButton(QToolButton):
    """Tool button that positions its menu above the button when possible."""

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.menu() is not None:
            self.setDown(True)
            self._popup_menu_upward()
            event.accept()
            return
        super().mousePressEvent(event)

    def _popup_menu_upward(self) -> None:
        menu = self.menu()
        if menu is None:
            return

        try:
            menu.aboutToHide.disconnect(self._on_menu_hidden)
        except (RuntimeError, TypeError):
            pass
        menu.aboutToHide.connect(self._on_menu_hidden)

        size = menu.sizeHint()
        above = self.mapToGlobal(QPoint(0, -size.height()))
        below = self.mapToGlobal(QPoint(0, self.height()))
        screen = QApplication.screenAt(self.mapToGlobal(self.rect().center()))
        if screen is None:
            menu.popup(above)
            return

        bounds = screen.availableGeometry()
        x = max(bounds.left(), min(above.x(), bounds.right() - size.width()))
        y = above.y() if above.y() >= bounds.top() else below.y()
        menu.popup(QPoint(x, y))

    def _on_menu_hidden(self) -> None:
        self.setDown(False)


class ViewportActionsBar(QFrame):
    """Floating bottom-center actions for canvas view and grid tools."""

    tool_selected = Signal(str)

    _MARGIN = 12
    _BTN_SIZE = 32

    def __init__(
        self,
        canvas,
        calibration_model=None,
        parent: QWidget = None,
        anomaly_constraint_model=None,
    ) -> None:
        super().__init__(parent or canvas)
        self._canvas = canvas
        self._model = None
        self._anomaly_model = None
        self._active_tool = ""
        self._has_image = False
        self._refreshing = False

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setAutoFillBackground(True)
        self.setObjectName("viewportActionsBar")
        self.setStyleSheet(
            """
            QFrame#viewportActionsBar {
                background: palette(window);
                border: 1px solid palette(mid);
                border-radius: 8px;
            }
            """
        )

        font = QFont()
        font.setPointSize(16)
        font.setBold(True)

        font_large = QFont()
        font_large.setPointSize(20)
        font_large.setBold(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        self._btn_zoom_in = self._make_button("+", "Zoom In")
        self._btn_zoom_in.setFont(font)
        self._btn_zoom_in.clicked.connect(canvas.zoom_in)
        layout.addWidget(self._btn_zoom_in)

        self._btn_zoom_out = self._make_button("-", "Zoom Out")
        self._btn_zoom_out.setFont(font)
        self._btn_zoom_out.clicked.connect(canvas.zoom_out)
        layout.addWidget(self._btn_zoom_out)

        self._btn_reset = self._make_button("⊡", "Reset View")
        self._btn_reset.setFont(font_large)
        self._btn_reset.clicked.connect(canvas.reset_view)
        layout.addWidget(self._btn_reset)

        self._add_divider(layout)

        self._btn_settings = self._make_popup_button("⊞", "Grid Settings")
        self._btn_settings.setFont(font_large)
        self._btn_settings.setMenu(self._build_settings_menu())
        layout.addWidget(self._btn_settings)

        self._btn_anomaly = self._make_popup_button("⊿", "Anomaly Constraints")
        font_anomaly = QFont()
        font_anomaly.setPointSize(13)
        font_anomaly.setBold(True)
        self._btn_anomaly.setFont(font_anomaly)
        self._btn_anomaly.setMenu(self._build_anomaly_menu())
        layout.addWidget(self._btn_anomaly)

        self.adjustSize()
        already_loaded = hasattr(canvas, "is_image_loaded") and canvas.is_image_loaded()
        self.set_image_loaded(already_loaded)
        if calibration_model is not None:
            self.set_calibration_model(calibration_model)
        else:
            self._refresh_controls()
        if anomaly_constraint_model is not None:
            self.set_anomaly_constraint_model(anomaly_constraint_model)

    def _make_button(self, text: str, tooltip: str) -> QToolButton:
        btn = QToolButton()
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setFixedSize(self._BTN_SIZE, self._BTN_SIZE)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def _make_popup_button(self, text: str, tooltip: str) -> QToolButton:
        btn = _UpwardMenuToolButton()
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setFixedSize(self._BTN_SIZE, self._BTN_SIZE)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def _add_divider(self, layout: QHBoxLayout) -> None:
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider)

    def _build_anomaly_menu(self) -> QMenu:
        menu = QMenu(self)
        action = QWidgetAction(self)
        panel = QWidget()
        panel.setMinimumWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        # ── Header + master enable ──────────────────────────────────────
        header = QLabel("Anomaly Constraints")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

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
        self._anomaly_area_color_btn.clicked.connect(
            self._on_anomaly_area_color_clicked
        )
        self._anomaly_dist_chk.toggled.connect(self._on_anomaly_dist_check_toggled)
        self._anomaly_dist_spin.valueChanged.connect(self._on_anomaly_dist_changed)
        self._anomaly_centroid_radio.toggled.connect(self._on_anomaly_method_changed)
        self._anomaly_dist_color_btn.clicked.connect(
            self._on_anomaly_dist_color_clicked
        )

        action.setDefaultWidget(panel)
        menu.addAction(action)
        return menu

    def _on_anomaly_enable_toggled(self, checked: bool) -> None:
        if self._anomaly_model is not None and not self._refreshing:
            self._anomaly_model.set_enabled(checked)

    def _on_anomaly_area_check_toggled(self, checked: bool) -> None:
        if self._anomaly_model is not None and not self._refreshing:
            self._anomaly_model.set_area_check_enabled(checked)

    def _on_anomaly_area_changed(self, value: float) -> None:
        if self._anomaly_model is not None and not self._refreshing:
            self._anomaly_model.set_area_threshold(value)

    def _on_anomaly_dist_check_toggled(self, checked: bool) -> None:
        if self._anomaly_model is not None and not self._refreshing:
            self._anomaly_model.set_distance_check_enabled(checked)

    def _on_anomaly_dist_changed(self, value: float) -> None:
        if self._anomaly_model is not None and not self._refreshing:
            self._anomaly_model.set_distance_threshold(value)

    def _on_anomaly_method_changed(self, centroid_checked: bool) -> None:
        if self._anomaly_model is not None and not self._refreshing:
            method = "centroid" if centroid_checked else "edge"
            self._anomaly_model.set_distance_method(method)

    def _on_anomaly_area_color_clicked(self) -> None:
        if self._anomaly_model is None:
            return
        r, g, b = self._anomaly_model.area_color()
        color = QColorDialog.getColor(QColor(r, g, b), self, "Area Violation Color")
        if color.isValid():
            self._anomaly_model.set_area_color(
                (color.red(), color.green(), color.blue())
            )

    def _on_anomaly_dist_color_clicked(self) -> None:
        if self._anomaly_model is None:
            return
        r, g, b = self._anomaly_model.distance_color()
        color = QColorDialog.getColor(QColor(r, g, b), self, "Proximity Line Color")
        if color.isValid():
            self._anomaly_model.set_distance_color(
                (color.red(), color.green(), color.blue())
            )

    def _apply_color_swatch(self, btn, rgb: tuple) -> None:
        r, g, b = rgb
        btn.setStyleSheet(
            f"QPushButton {{ background-color: rgb({r},{g},{b}); "
            f"border: 1px solid #888; border-radius: 2px; }}"
        )

    def _refresh_anomaly_controls(self) -> None:
        if not hasattr(self, "_anomaly_enable_chk"):
            return
        if self._anomaly_model is None:
            return
        self._refreshing = True
        self._anomaly_enable_chk.setChecked(self._anomaly_model.enabled())
        self._anomaly_area_chk.setChecked(self._anomaly_model.area_check_enabled())
        self._anomaly_area_spin.setValue(self._anomaly_model.area_threshold())
        self._apply_color_swatch(
            self._anomaly_area_color_btn, self._anomaly_model.area_color()
        )
        self._anomaly_dist_chk.setChecked(self._anomaly_model.distance_check_enabled())
        self._anomaly_dist_spin.setValue(self._anomaly_model.distance_threshold())
        self._anomaly_centroid_radio.setChecked(
            self._anomaly_model.distance_method() == "centroid"
        )
        self._anomaly_edge_radio.setChecked(
            self._anomaly_model.distance_method() == "edge"
        )
        self._apply_color_swatch(
            self._anomaly_dist_color_btn, self._anomaly_model.distance_color()
        )
        self._refreshing = False

    def _build_settings_menu(self) -> QMenu:
        menu = QMenu(self)
        action = QWidgetAction(self)
        panel = QWidget()
        panel.setMinimumWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        # ═══════════════════════════════════════════════════════════════
        # CALIBRATION SECTION
        # ═══════════════════════════════════════════════════════════════
        calib_header = QLabel("Calibration")
        calib_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(calib_header)

        self._calib_status_lbl = QLabel("Current Calibration: None")
        layout.addWidget(self._calib_status_lbl)

        # Ratio input: [ 1 ] px : [ 0.05 ] [ mm▾ ] [ Apply ]
        ratio_row = QHBoxLayout()
        ratio_row.setSpacing(4)
        self._ratio_px_spin = QSpinBox()
        self._ratio_px_spin.setRange(1, 999999)
        self._ratio_px_spin.setValue(1)
        self._ratio_px_spin.setToolTip("Number of pixels on the left side of the ratio")
        ratio_row.addWidget(self._ratio_px_spin)
        ratio_row.addWidget(QLabel("px :"))
        self._ratio_val_num_spin = QDoubleSpinBox()
        self._ratio_val_num_spin.setRange(0.0, 999999.0)
        self._ratio_val_num_spin.setDecimals(2)
        self._ratio_val_num_spin.setSingleStep(0.1)
        self._ratio_val_num_spin.setToolTip(
            "Real-world value for the right side of the ratio"
        )
        ratio_row.addWidget(self._ratio_val_num_spin)
        self._ratio_unit_combo = QComboBox()
        for _u in ("mm", "um", "nm", "pm", "fm", "cm", "dm", "m", "km"):
            self._ratio_unit_combo.addItem(_u)
        self._ratio_unit_combo.setFixedWidth(52)
        ratio_row.addWidget(self._ratio_unit_combo)
        self._btn_apply_ratio = QPushButton("Apply")
        self._btn_apply_ratio.setFixedWidth(50)
        self._btn_apply_ratio.clicked.connect(self._on_apply_ratio_clicked)
        ratio_row.addWidget(self._btn_apply_ratio)
        layout.addLayout(ratio_row)

        # Import / Export calibration file buttons
        ratio_file_row = QHBoxLayout()
        ratio_file_row.setSpacing(4)
        self._btn_import_ratio = QPushButton("Import")
        self._btn_import_ratio.setToolTip("Load a ratio from a plain-text .txt file")
        self._btn_import_ratio.clicked.connect(self._on_import_ratio_clicked)
        ratio_file_row.addWidget(self._btn_import_ratio)
        self._btn_export_ratio = QPushButton("Export")
        self._btn_export_ratio.setToolTip(
            "Save the current ratio to a plain-text .txt file"
        )
        self._btn_export_ratio.setEnabled(False)  # enabled once a calibration is set
        self._btn_export_ratio.clicked.connect(self._on_export_ratio_clicked)
        ratio_file_row.addWidget(self._btn_export_ratio)
        layout.addLayout(ratio_file_row)

        self._btn_calibrate_points = QPushButton("✛  Click two points…")
        self._btn_calibrate_points.setCheckable(True)
        self._btn_calibrate_points.setToolTip(
            "Click two known points on the image, then enter the real distance"
        )
        self._btn_calibrate_points.clicked.connect(
            lambda checked: self._on_tool_clicked("calibrate", checked)
        )
        layout.addWidget(self._btn_calibrate_points)

        # Measurement result
        meas_row = QHBoxLayout()
        meas_row.setSpacing(4)
        self._meas_lbl = QLabel("Distance: -")
        self._meas_lbl.setStyleSheet("font-weight: bold;")
        meas_row.addWidget(self._meas_lbl)
        meas_row.addStretch()
        self._btn_clear_measurement = QPushButton("✕")
        self._btn_clear_measurement.setFixedSize(24, 24)
        self._btn_clear_measurement.setToolTip("Clear measurement")
        self._btn_clear_measurement.clicked.connect(self._on_clear_measurement_clicked)
        meas_row.addWidget(self._btn_clear_measurement)
        layout.addLayout(meas_row)

        self._btn_reset_calibration = QPushButton("Reset to pixels")
        self._btn_reset_calibration.setToolTip(
            "Remove calibration and return to pixel units"
        )
        self._btn_reset_calibration.clicked.connect(self._on_reset_calibration_clicked)
        layout.addWidget(self._btn_reset_calibration)

        # ── Section divider ───────────────────────────────────────────
        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setFrameShadow(QFrame.Sunken)
        layout.addWidget(div)

        # ═══════════════════════════════════════════════════════════════
        # GRID SECTION
        # ═══════════════════════════════════════════════════════════════
        grid_header = QLabel("Grid")
        grid_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(grid_header)

        self._grid_chk = QCheckBox("Show Grid")
        self._grid_chk.toggled.connect(self._on_grid_toggled)
        layout.addWidget(self._grid_chk)

        opacity_row = QHBoxLayout()
        opacity_row.setSpacing(6)
        opacity_row.addWidget(QLabel("Opacity"))
        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(50)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self._opacity_slider)
        self._opacity_lbl = QLabel("50%")
        self._opacity_lbl.setFixedWidth(34)
        opacity_row.addWidget(self._opacity_lbl)
        layout.addLayout(opacity_row)

        spacing_mode_row = QHBoxLayout()
        spacing_mode_row.setSpacing(6)
        spacing_mode_row.addWidget(QLabel("Spacing"))
        spacing_mode_row.addStretch()
        self._radio_auto = QRadioButton("Auto")
        self._radio_fixed = QRadioButton("Fixed")
        self._radio_auto.setChecked(True)
        spacing_group = QButtonGroup(self)
        spacing_group.addButton(self._radio_auto)
        spacing_group.addButton(self._radio_fixed)
        self._radio_auto.toggled.connect(self._on_spacing_mode_changed)
        spacing_mode_row.addWidget(self._radio_auto)
        spacing_mode_row.addWidget(self._radio_fixed)
        layout.addLayout(spacing_mode_row)

        spacing_val_row = QHBoxLayout()
        spacing_val_row.setSpacing(6)
        self._spacing_edit = QLineEdit()
        self._spacing_edit.setPlaceholderText("1.0")
        self._spacing_edit.setEnabled(False)
        self._spacing_edit.editingFinished.connect(self._on_spacing_edited)
        spacing_val_row.addWidget(self._spacing_edit)
        self._unit_lbl = QLabel("px")
        spacing_val_row.addWidget(self._unit_lbl)
        layout.addLayout(spacing_val_row)

        color_row = QHBoxLayout()
        color_row.setSpacing(6)
        color_row.addWidget(QLabel("Color"))
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(32, 20)
        self._color_btn.setToolTip("Change grid color")
        self._color_btn.clicked.connect(self._on_color_clicked)
        color_row.addWidget(self._color_btn)
        color_row.addStretch()
        layout.addLayout(color_row)

        self._update_color_swatch((58, 90, 122))
        action.setDefaultWidget(panel)
        menu.addAction(action)
        return menu

    def set_calibration_model(self, model) -> None:
        self._model = model
        model.calibration_changed.connect(self._refresh_all)
        model.grid_changed.connect(self._refresh_all)
        model.measurement_updated.connect(self._refresh_measurement)
        self._refresh_all()

    def set_anomaly_constraint_model(self, model) -> None:
        self._anomaly_model = model
        model.constraints_changed.connect(self._refresh_anomaly_controls)
        self._refresh_anomaly_controls()

    def refresh_anomaly_violations(self, area_count: int, dist_count: int) -> None:
        """Update the inline violation count labels with threshold-aware text."""
        if not hasattr(self, "_anomaly_area_count_lbl"):
            return

        if area_count > 0 and self._anomaly_model is not None:
            t = self._anomaly_model.area_threshold()
            unit = self._anomaly_area_unit_lbl.text()
            label = "defect" if area_count == 1 else "defects"
            self._anomaly_area_count_lbl.setText(f"{area_count} {label} > {t:g}{unit}")
        else:
            self._anomaly_area_count_lbl.setText("")

        if dist_count > 0 and self._anomaly_model is not None:
            t = self._anomaly_model.distance_threshold()
            unit = self._anomaly_dist_unit_lbl.text()
            label = "defect" if dist_count == 1 else "defects"
            self._anomaly_dist_count_lbl.setText(
                f"{dist_count} {label} within {t:g}{unit}"
            )
        else:
            self._anomaly_dist_count_lbl.setText("")

    def update_anomaly_units(self, unit: str) -> None:
        """Update the unit labels in the anomaly panel (called when calibration changes)."""
        if not hasattr(self, "_anomaly_area_unit_lbl"):
            return
        self._anomaly_area_unit_lbl.setText(f"{unit}²")
        self._anomaly_dist_unit_lbl.setText(unit)

    def set_image_loaded(self, loaded: bool) -> None:
        self._has_image = loaded
        self._refresh_action_availability()

    def set_active_tool(self, tool_name: str) -> None:
        self._active_tool = tool_name if tool_name == "calibrate" else ""
        self._refreshing = True
        self._btn_calibrate_points.setChecked(self._active_tool == "calibrate")
        self._refreshing = False

    def toggle_calibrate(self) -> None:
        if self._btn_calibrate_points.isEnabled():
            self._on_tool_clicked("calibrate", self._active_tool != "calibrate")

    def reposition(self, canvas_size) -> None:
        self.adjustSize()
        x = (canvas_size.width() - self.width()) // 2
        y = canvas_size.height() - self.height() - self._MARGIN
        self.move(max(0, x), max(0, y))
        self._canvas.set_watermark_bar_y(max(0, y))

    def _on_tool_clicked(self, tool_name: str, checked: bool) -> None:
        if self._refreshing:
            return
        self._active_tool = tool_name if checked else ""
        self.set_active_tool(self._active_tool)
        self.tool_selected.emit(self._active_tool)

    def _on_apply_ratio_clicked(self) -> None:
        if self._model is None:
            return
        px_count = self._ratio_px_spin.value()
        world_val = self._ratio_val_num_spin.value()
        unit = self._ratio_unit_combo.currentText()
        if world_val <= 0:
            QMessageBox.warning(
                self, "Invalid Ratio", "World value must be greater than zero."
            )
            return
        self._model.apply_scale_direct(px_count, world_val, unit)

    def _on_import_ratio_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Calibration Ratio",
            "",
            "Calibration Ratio (*.txt)",
        )
        if not path:
            return
        from core.persistence.calibration_io import read_calibration_ratio

        try:
            data = read_calibration_ratio(path)
            self._model.apply_scale_direct(
                data["px_count"], data["world_val"], data["unit"]
            )
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", str(exc))

    def _on_export_ratio_clicked(self) -> None:
        if self._model is None or not self._model.has_scale():
            return  # button is disabled in this state; guard kept as a safety net
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Calibration Ratio",
            "calibration.txt",
            "Calibration Ratio (*.txt)",
        )
        if not path:
            return
        from core.persistence.calibration_io import write_calibration_ratio

        try:
            write_calibration_ratio(
                path,
                self._model.px_count(),
                self._model.world_val(),
                self._model.unit(),
            )
            QMessageBox.information(
                self, "Export Calibration Ratio", f"Saved to:\n{path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _on_grid_toggled(self, checked: bool) -> None:
        if self._model is not None and not self._refreshing:
            self._model.set_grid_visible(checked)

    def _on_opacity_changed(self, value: int) -> None:
        self._opacity_lbl.setText(f"{value}%")
        if self._model is not None and not self._refreshing:
            self._model.set_grid_opacity(value / 100.0)

    def _on_color_clicked(self) -> None:
        if self._model is None:
            return
        r, g, b = self._model.grid_color()
        color = QColorDialog.getColor(QColor(r, g, b), self, "Grid Color")
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            self._model.set_grid_color(rgb)
            self._update_color_swatch(rgb)

    def _on_spacing_mode_changed(self, auto_checked: bool) -> None:
        self._spacing_edit.setEnabled(not auto_checked)
        if self._model is None or self._refreshing:
            return
        if auto_checked:
            self._model.set_grid_spacing_auto()
        else:
            self._try_apply_spacing()

    def _on_spacing_edited(self) -> None:
        if (
            self._model is not None
            and not self._refreshing
            and self._radio_fixed.isChecked()
        ):
            self._try_apply_spacing()

    def _on_clear_measurement_clicked(self) -> None:
        if self._model is not None:
            self._model.clear_measurement()

    def _on_reset_calibration_clicked(self) -> None:
        if self._model is not None:
            self._model.clear_calibration()

    def _try_apply_spacing(self) -> None:
        try:
            value = float(self._spacing_edit.text())
        except ValueError:
            return
        if value > 0:
            self._model.set_grid_spacing(value)

    def _refresh_all(self) -> None:
        self._refresh_calib_status()
        self._refresh_controls()
        self._refresh_measurement()
        self._refresh_anomaly_controls()
        self._refresh_action_availability()

    def _refresh_calib_status(self) -> None:
        has_scale = self._model is not None and self._model.has_scale()
        self._btn_export_ratio.setEnabled(has_scale)
        if not has_scale:
            self._calib_status_lbl.setText("Current Calibration: None")
            return
        from core.persistence.calibration_io import format_ratio_string

        px_count = self._model.px_count()
        world_val = self._model.world_val()
        unit = self._model.unit()
        ratio_str = format_ratio_string(px_count, world_val, unit)
        self._calib_status_lbl.setText(f"Current Calibration: {ratio_str}")
        if not self._ratio_px_spin.hasFocus():
            self._ratio_px_spin.setValue(int(round(px_count)))
        if not self._ratio_val_num_spin.hasFocus():
            self._ratio_val_num_spin.setValue(world_val)
            idx = self._ratio_unit_combo.findText(unit)
            if idx >= 0:
                self._ratio_unit_combo.setCurrentIndex(idx)
            else:
                self._ratio_unit_combo.addItem(unit)
                self._ratio_unit_combo.setCurrentText(unit)

    def _refresh_controls(self) -> None:
        if self._model is None:
            self._refresh_action_availability()
            return
        self._refreshing = True
        grid_visible = self._model.grid_visible() and self._model.has_scale()
        self._grid_chk.setChecked(grid_visible)
        opacity_pct = int(self._model.grid_opacity() * 100)
        self._opacity_slider.setValue(opacity_pct)
        self._opacity_lbl.setText(f"{opacity_pct}%")
        self._update_color_swatch(self._model.grid_color())
        auto_spacing = self._model.grid_spacing_auto()
        self._radio_auto.setChecked(auto_spacing)
        self._radio_fixed.setChecked(not auto_spacing)
        self._spacing_edit.setEnabled(not auto_spacing)
        self._spacing_edit.setText(f"{self._model.grid_spacing_world():g}")
        self._unit_lbl.setText(self._model.unit())
        self._refreshing = False
        self._refresh_action_availability()

    def _refresh_measurement(self) -> None:
        if self._model is None:
            self._meas_lbl.setText("Distance: -")
            return
        dist = self._model.measured_distance()
        if dist is None:
            p1, p2 = self._model.meas_points()
            if p1 is not None and p2 is None:
                self._meas_lbl.setText("Distance: click point B...")
            else:
                self._meas_lbl.setText("Distance: -")
            return
        self._meas_lbl.setText(f"Distance: {dist:.1f} {self._model.unit()}")

    def _refresh_action_availability(self) -> None:
        scale_available = self._model is not None and self._model.has_scale()
        self._btn_zoom_in.setEnabled(self._has_image)
        self._btn_zoom_out.setEnabled(self._has_image)
        self._btn_reset.setEnabled(self._has_image)
        self._btn_settings.setEnabled(self._has_image)
        self._btn_anomaly.setEnabled(self._has_image)
        self._btn_calibrate_points.setEnabled(self._has_image)
        self._btn_import_ratio.setEnabled(True)
        self._btn_export_ratio.setEnabled(scale_available)
        self._btn_apply_ratio.setEnabled(True)
        self._grid_chk.setEnabled(scale_available)
        self._opacity_slider.setEnabled(scale_available)
        self._color_btn.setEnabled(scale_available)
        self._radio_auto.setEnabled(scale_available)
        self._radio_fixed.setEnabled(scale_available)
        self._spacing_edit.setEnabled(scale_available and self._radio_fixed.isChecked())
        self._btn_clear_measurement.setEnabled(scale_available)
        self._btn_reset_calibration.setEnabled(scale_available)

    def _update_color_swatch(self, rgb: tuple) -> None:
        r, g, b = rgb
        self._color_btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #888;"
        )
