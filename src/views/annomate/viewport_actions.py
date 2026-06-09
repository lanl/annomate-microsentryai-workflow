from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
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
    center_calibration_started = Signal()
    center_calibration_accepted = Signal()
    center_template_cleared = Signal()
    crop_overlay_toggled = Signal(bool)

    _MARGIN = 12
    _BTN_SIZE = 32

    def __init__(
        self,
        canvas,
        calibration_model=None,
        parent: QWidget = None,
        center_template_model=None,
    ) -> None:
        super().__init__(parent or canvas)
        self._canvas = canvas
        self._model = None
        self._center_template_model = None
        self._active_tool = ""
        self._has_image = False
        self._image_w = 0
        self._image_h = 0
        self._refreshing = False
        self._center_calibrating = False

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

        self._btn_reset = self._make_button("⊙", "Reset View")
        self._btn_reset.setFont(font)
        self._btn_reset.clicked.connect(canvas.reset_view)
        layout.addWidget(self._btn_reset)

        self._add_divider(layout)

        self._btn_measure = self._make_button("⇔", "Measure Distance (M)")
        self._btn_measure.setCheckable(True)
        self._btn_measure.setFont(font)
        self._btn_measure.clicked.connect(
            lambda checked: self._on_tool_clicked("measure", checked)
        )
        layout.addWidget(self._btn_measure)

        self._btn_settings = self._make_popup_button("⚙", "Grid Settings")
        self._btn_settings.setFont(font)
        self._btn_settings.setMenu(self._build_settings_menu())
        layout.addWidget(self._btn_settings)

        self._add_divider(layout)

        self._btn_crop = self._make_popup_button("⌗", "Center Crop")
        self._btn_crop.setFont(font)
        self._btn_crop.setMenu(self._build_crop_menu())
        layout.addWidget(self._btn_crop)

        self.adjustSize()
        self.set_image_loaded(False)
        if hasattr(canvas, "image_loaded"):
            canvas.image_loaded.connect(self.set_image_dimensions)
        if hasattr(canvas, "centerCropChanged"):
            canvas.centerCropChanged.connect(lambda _: self._refresh_crop_controls())
        if calibration_model is not None:
            self.set_calibration_model(calibration_model)
        else:
            self._refresh_controls()
        if center_template_model is not None:
            self.set_center_template_model(center_template_model)

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

        # Ratio input: [ 1px ] : [ 0.05mm ] [ Apply ]
        ratio_row = QHBoxLayout()
        ratio_row.setSpacing(4)
        self._ratio_px_edit = QLineEdit()
        self._ratio_px_edit.setPlaceholderText("1px")
        self._ratio_px_edit.setToolTip("Left side of ratio, e.g. 1px or 50px")
        ratio_row.addWidget(self._ratio_px_edit)
        ratio_row.addWidget(QLabel(":"))
        self._ratio_val_edit = QLineEdit()
        self._ratio_val_edit.setPlaceholderText("0.05mm")
        self._ratio_val_edit.setToolTip(
            "Right side of ratio, e.g. 0.05mm, 100um, 1furlong"
        )
        ratio_row.addWidget(self._ratio_val_edit)
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

    def _build_crop_menu(self) -> QMenu:
        menu = QMenu(self)
        action = QWidgetAction(self)
        panel = QWidget()
        panel.setMinimumWidth(260)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 8, 10, 8)
        panel_layout.setSpacing(8)

        # Header
        header = QLabel("Center Crop")
        header.setStyleSheet("font-weight: bold;")
        panel_layout.addWidget(header)

        # Enable + Shape on one row
        enable_shape_row = QHBoxLayout()
        enable_shape_row.setSpacing(8)
        self._crop_chk = QCheckBox("Enable")
        self._crop_chk.toggled.connect(self._on_crop_toggled)
        enable_shape_row.addWidget(self._crop_chk)
        enable_shape_row.addStretch()
        enable_shape_row.addWidget(QLabel("Shape"))
        self._crop_shape_combo = QComboBox()
        self._crop_shape_combo.addItems(["Rectangle", "Circle"])
        self._crop_shape_combo.currentTextChanged.connect(self._on_crop_shape_changed)
        enable_shape_row.addWidget(self._crop_shape_combo)
        panel_layout.addLayout(enable_shape_row)

        # Width
        width_row = QHBoxLayout()
        width_row.setSpacing(8)
        self._crop_primary_lbl = QLabel("Width")
        self._crop_primary_lbl.setFixedWidth(44)
        width_row.addWidget(self._crop_primary_lbl)
        self._crop_width_spin = QSpinBox()
        self._crop_width_spin.setRange(1, 999999)
        self._crop_width_spin.setSuffix(" px")
        self._crop_width_spin.valueChanged.connect(self._on_crop_primary_changed)
        width_row.addWidget(self._crop_width_spin)
        panel_layout.addLayout(width_row)

        # Height
        height_row = QHBoxLayout()
        height_row.setSpacing(8)
        self._crop_secondary_lbl = QLabel("Height")
        self._crop_secondary_lbl.setFixedWidth(44)
        height_row.addWidget(self._crop_secondary_lbl)
        self._crop_height_spin = QSpinBox()
        self._crop_height_spin.setRange(1, 999999)
        self._crop_height_spin.setSuffix(" px")
        self._crop_height_spin.valueChanged.connect(self._on_crop_secondary_changed)
        height_row.addWidget(self._crop_height_spin)
        panel_layout.addLayout(height_row)

        # Outside opacity
        opacity_row = QHBoxLayout()
        opacity_row.setSpacing(8)
        opacity_row.addWidget(QLabel("Outside opacity"))
        self._crop_opacity_slider = QSlider(Qt.Horizontal)
        self._crop_opacity_slider.setRange(0, 100)
        self._crop_opacity_slider.valueChanged.connect(self._on_crop_opacity_changed)
        opacity_row.addWidget(self._crop_opacity_slider)
        self._crop_opacity_lbl = QLabel("37%")
        self._crop_opacity_lbl.setFixedWidth(34)
        opacity_row.addWidget(self._crop_opacity_lbl)
        panel_layout.addLayout(opacity_row)

        # Center dot
        self._crop_center_dot_chk = QCheckBox("Show center dot")
        self._crop_center_dot_chk.setToolTip("Show center dot")
        self._crop_center_dot_chk.toggled.connect(self._on_crop_center_dot_toggled)
        panel_layout.addWidget(self._crop_center_dot_chk)

        # Hint
        self._crop_hint_lbl = QLabel("Centered on image")
        self._crop_hint_lbl.setStyleSheet("color: grey; font-style: italic;")
        panel_layout.addWidget(self._crop_hint_lbl)

        # Template divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        panel_layout.addWidget(divider)

        template_header = QLabel("Template")
        template_header.setStyleSheet("font-weight: bold;")
        panel_layout.addWidget(template_header)

        self._template_status_lbl = QLabel("Template: none")
        self._template_status_lbl.setWordWrap(True)
        self._template_status_lbl.setStyleSheet("color: grey; font-style: italic;")
        panel_layout.addWidget(self._template_status_lbl)

        self._btn_calibrate_center = QPushButton("Calibrate Center")
        self._btn_calibrate_center.setToolTip("Move the crop and dot together")
        self._btn_calibrate_center.clicked.connect(self._on_calibrate_center_clicked)
        panel_layout.addWidget(self._btn_calibrate_center)

        accept_clear_row = QHBoxLayout()
        accept_clear_row.setSpacing(6)
        self._btn_accept_center = QPushButton("Accept")
        self._btn_accept_center.setToolTip("Save this center as the matching template")
        self._btn_accept_center.clicked.connect(self._on_accept_center_clicked)
        accept_clear_row.addWidget(self._btn_accept_center)
        self._btn_clear_template = QPushButton("Clear Template")
        self._btn_clear_template.setToolTip("Clear saved center template")
        self._btn_clear_template.clicked.connect(self._on_clear_template_clicked)
        accept_clear_row.addWidget(self._btn_clear_template)
        panel_layout.addLayout(accept_clear_row)

        # Reset at the bottom
        self._btn_reset_crop = QPushButton("Reset Defaults")
        self._btn_reset_crop.setToolTip(
            "Reset crop shape, size, and opacity to defaults"
        )
        self._btn_reset_crop.clicked.connect(self._on_reset_crop_clicked)
        panel_layout.addWidget(self._btn_reset_crop)

        action.setDefaultWidget(panel)
        menu.addAction(action)
        return menu

    def set_calibration_model(self, model) -> None:
        self._model = model
        model.calibration_changed.connect(self._refresh_all)
        model.grid_changed.connect(self._refresh_all)
        model.measurement_updated.connect(self._refresh_measurement)
        self._refresh_all()

    def set_center_template_model(self, model) -> None:
        self._center_template_model = model
        model.template_changed.connect(self._refresh_template_status)
        model.match_changed.connect(self._refresh_template_status)
        self._refresh_template_status()
        self._refresh_action_availability()

    def set_image_loaded(self, loaded: bool) -> None:
        self._has_image = loaded
        self._refresh_action_availability()

    def set_image_dimensions(self, width: int, height: int) -> None:
        self._image_w = max(0, int(width))
        self._image_h = max(0, int(height))
        self._has_image = self._image_w > 0 and self._image_h > 0
        self._refresh_crop_controls()
        self._refresh_action_availability()

    def set_active_tool(self, tool_name: str) -> None:
        self._active_tool = tool_name if tool_name in ("calibrate", "measure") else ""
        self._refreshing = True
        self._btn_calibrate_points.setChecked(self._active_tool == "calibrate")
        self._btn_measure.setChecked(self._active_tool == "measure")
        self._refreshing = False

    def toggle_calibrate(self) -> None:
        if self._btn_calibrate_points.isEnabled():
            self._on_tool_clicked("calibrate", self._active_tool != "calibrate")

    def toggle_measure(self) -> None:
        if self._btn_measure.isEnabled():
            self._on_tool_clicked("measure", self._active_tool != "measure")

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
        px_text = self._ratio_px_edit.text().strip()
        val_text = self._ratio_val_edit.text().strip()
        if not px_text or not val_text:
            return
        from core.persistence.calibration_io import parse_ratio_string

        try:
            px_count, world_val, unit = parse_ratio_string(f"{px_text}:{val_text}")
            self._model.apply_scale_direct(px_count, world_val, unit)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Ratio", str(exc))

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
            QMessageBox.warning(self, "Export Calibration Ratio", "No calibration set.")
            return
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

    def _on_crop_toggled(self, checked: bool) -> None:
        if self._refreshing:
            return
        if not checked:
            self._center_calibrating = False
        self._canvas.set_center_crop(
            enabled=checked,
            calibrating=False if not checked else None,
        )
        self._refresh_crop_controls()
        self.crop_overlay_toggled.emit(checked)

    def _on_crop_shape_changed(self, display_name: str) -> None:
        if self._refreshing:
            return
        shape = display_name.lower()
        if shape == "circle":
            diameter = min(
                self._crop_width_spin.value(),
                self._crop_height_spin.value(),
            )
            self._canvas.set_center_crop(
                shape=shape,
                width=diameter,
                height=diameter,
            )
        else:
            self._canvas.set_center_crop(shape=shape)
        self._refresh_crop_controls()

    def _on_crop_primary_changed(self, value: int) -> None:
        if self._refreshing:
            return
        if self._crop_shape_combo.currentText() == "Circle":
            diameter = max(1, value)
            self._canvas.set_center_crop(width=diameter, height=diameter)
        else:
            self._canvas.set_center_crop(
                width=value,
                height=self._crop_height_spin.value(),
            )
        self._refresh_crop_controls()

    def _on_crop_secondary_changed(self, value: int) -> None:
        if self._refreshing:
            return
        if self._crop_shape_combo.currentText() == "Circle":
            diameter = max(1, value * 2)
            self._canvas.set_center_crop(width=diameter, height=diameter)
        else:
            self._canvas.set_center_crop(
                width=self._crop_width_spin.value(),
                height=value,
            )
        self._refresh_crop_controls()

    def _on_crop_opacity_changed(self, value: int) -> None:
        self._crop_opacity_lbl.setText(f"{value}%")
        if self._refreshing:
            return
        self._canvas.set_center_crop(opacity=value / 100.0)

    def _on_crop_center_dot_toggled(self, checked: bool) -> None:
        if self._refreshing:
            return
        self._canvas.set_center_crop(center_dot=checked)

    def _on_reset_crop_clicked(self) -> None:
        if self._refreshing:
            return
        self._center_calibrating = False
        self._canvas.set_center_crop(
            shape="circle",
            width=1210,
            height=1210,
            opacity=0.37,
            center_dot=False,
            calibrating=False,
        )
        self._refresh_crop_controls()

    def _on_calibrate_center_clicked(self) -> None:
        if self._refreshing:
            return
        self.center_calibration_started.emit()

    def _on_accept_center_clicked(self) -> None:
        if self._refreshing:
            return
        self.center_calibration_accepted.emit()

    def _on_clear_template_clicked(self) -> None:
        if self._refreshing:
            return
        self.center_template_cleared.emit()

    def set_center_calibrating(self, active: bool) -> None:
        self._center_calibrating = bool(active)
        self._refresh_template_status()
        self._refresh_action_availability()

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
        self._refresh_crop_controls()
        self._refresh_action_availability()

    def _refresh_calib_status(self) -> None:
        if self._model is None or not self._model.has_scale():
            self._calib_status_lbl.setText("Current Calibration: None")
            if not self._ratio_px_edit.hasFocus():
                self._ratio_px_edit.clear()
            if not self._ratio_val_edit.hasFocus():
                self._ratio_val_edit.clear()
            return
        from core.persistence.calibration_io import format_ratio_string

        px_count = self._model.px_count()
        world_val = self._model.world_val()
        unit = self._model.unit()
        ratio_str = format_ratio_string(px_count, world_val, unit)
        self._calib_status_lbl.setText(f"Current Calibration: {ratio_str}")
        left, right = ratio_str.split(":", 1)
        if not self._ratio_px_edit.hasFocus():
            self._ratio_px_edit.setText(left)
        if not self._ratio_val_edit.hasFocus():
            self._ratio_val_edit.setText(right)

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

    def _refresh_template_status(self) -> None:
        if not hasattr(self, "_template_status_lbl"):
            return
        if self._center_calibrating:
            self._template_status_lbl.setText("Template: move crop, then Accept")
            self._template_status_lbl.setStyleSheet("color: black; font-style: normal;")
            self._refresh_action_availability()
            return
        model = self._center_template_model
        if model is None or not model.has_template():
            self._template_status_lbl.setText("Template: none")
            self._template_status_lbl.setStyleSheet("color: grey; font-style: italic;")
            self._refresh_action_availability()
            return
        score = model.last_score()
        if score is None:
            self._template_status_lbl.setText("Template: saved")
        else:
            self._template_status_lbl.setText(f"Template match: {score:.3f}")
        self._template_status_lbl.setStyleSheet("color: black; font-style: normal;")
        self._refresh_action_availability()

    def _refresh_crop_controls(self) -> None:
        settings = self._canvas.center_crop_settings()
        max_w = max(1, self._image_w)
        max_h = max(1, self._image_h)
        width = settings.get("width") or max_w // 2
        height = settings.get("height") or max_h // 2
        shape = settings.get("shape") or "rectangle"
        opacity_pct = int(round((settings.get("opacity") or 0.0) * 100))
        center_dot = bool(settings.get("center_dot"))

        self._refreshing = True
        self._crop_chk.setChecked(bool(settings.get("enabled")))
        if shape == "circle":
            max_diameter = max(1210, min(max_w, max_h))
            diameter = max(1, min(int(min(width, height)), max_diameter))
            self._crop_primary_lbl.setText("Diameter")
            self._crop_secondary_lbl.setText("Radius")
            self._crop_width_spin.setToolTip("Circle diameter")
            self._crop_height_spin.setToolTip("Circle radius")
            self._crop_width_spin.setRange(1, max_diameter)
            self._crop_height_spin.setRange(1, max(1, max_diameter // 2))
            self._crop_width_spin.setValue(diameter)
            self._crop_height_spin.setValue(max(1, diameter // 2))
        else:
            self._crop_primary_lbl.setText("W")
            self._crop_secondary_lbl.setText("H")
            self._crop_width_spin.setToolTip("Rectangle width")
            self._crop_height_spin.setToolTip("Rectangle height")
            self._crop_width_spin.setRange(1, max(1210, max_w))
            self._crop_height_spin.setRange(1, max(1210, max_h))
            self._crop_width_spin.setValue(
                max(1, min(int(width), self._crop_width_spin.maximum()))
            )
            self._crop_height_spin.setValue(
                max(1, min(int(height), self._crop_height_spin.maximum()))
            )
        shape_label = {
            "rectangle": "Rectangle",
            "circle": "Circle",
        }.get(shape, "Rectangle")
        self._crop_shape_combo.setCurrentText(shape_label)
        self._crop_opacity_slider.setValue(opacity_pct)
        self._crop_opacity_lbl.setText(f"{opacity_pct}%")
        self._crop_center_dot_chk.setChecked(center_dot)
        if settings.get("calibrating") != self._center_calibrating:
            self._center_calibrating = bool(settings.get("calibrating"))
        self._crop_hint_lbl.setText(
            (
                f"Center: {settings.get('center_x'):.0f}, {settings.get('center_y'):.0f}"
                if settings.get("center_x") is not None
                else f"Image: {self._image_w} × {self._image_h} px"
            )
            if self._has_image
            else "Load an image to preview a crop"
        )
        self._refreshing = False
        self._refresh_template_status()

    def _refresh_action_availability(self) -> None:
        scale_available = self._model is not None and self._model.has_scale()
        self._btn_calibrate_points.setEnabled(self._has_image)
        self._btn_import_ratio.setEnabled(True)
        self._btn_export_ratio.setEnabled(scale_available)
        self._btn_apply_ratio.setEnabled(True)
        self._btn_crop.setEnabled(self._has_image)
        self._crop_chk.setEnabled(self._has_image)
        self._crop_shape_combo.setEnabled(self._has_image)
        self._crop_width_spin.setEnabled(self._has_image)
        self._crop_height_spin.setEnabled(self._has_image)
        self._crop_opacity_slider.setEnabled(self._has_image)
        self._crop_center_dot_chk.setEnabled(self._has_image)
        self._btn_reset_crop.setEnabled(self._has_image)
        self._btn_calibrate_center.setEnabled(self._has_image)
        self._btn_accept_center.setEnabled(self._has_image and self._center_calibrating)
        self._btn_clear_template.setEnabled(
            self._center_template_model is not None
            and self._center_template_model.has_template()
        )
        self._btn_measure.setEnabled(scale_available and self._has_image)
        self._grid_chk.setEnabled(scale_available)
        self._opacity_slider.setEnabled(scale_available)
        self._color_btn.setEnabled(scale_available)
        self._radio_auto.setEnabled(scale_available)
        self._radio_fixed.setEnabled(scale_available)
        self._spacing_edit.setEnabled(scale_available and self._radio_fixed.isChecked())
        self._btn_clear_measurement.setEnabled(scale_available)
        self._btn_reset_calibration.setEnabled(scale_available)
        if not scale_available and self._active_tool == "measure":
            self.set_active_tool("")
            self.tool_selected.emit("")

    def _update_color_swatch(self, rgb: tuple) -> None:
        r, g, b = rgb
        self._color_btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #888;"
        )
