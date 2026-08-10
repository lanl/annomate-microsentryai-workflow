from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
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
    """Floating bottom-center actions for canvas view and calibration tools."""

    tool_selected = Signal(str)

    _MARGIN = 12
    _BTN_SIZE = 32

    def __init__(
        self,
        canvas,
        calibration_model=None,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent or canvas)
        self._canvas = canvas
        self._model = None
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

        self._btn_settings = self._make_popup_button("⊞", "Calibration")
        self._btn_settings.setFont(font_large)
        self._btn_settings.setMenu(self._build_settings_menu())
        layout.addWidget(self._btn_settings)

        self.adjustSize()
        already_loaded = hasattr(canvas, "is_image_loaded") and canvas.is_image_loaded()
        self.set_image_loaded(already_loaded)
        if calibration_model is not None:
            self.set_calibration_model(calibration_model)
        else:
            self._refresh_action_availability()

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

        action.setDefaultWidget(panel)
        menu.addAction(action)
        return menu

    def set_calibration_model(self, model) -> None:
        self._model = model
        model.calibration_changed.connect(self._refresh_all)
        model.measurement_updated.connect(self._refresh_measurement)
        self._refresh_all()


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

    def _on_clear_measurement_clicked(self) -> None:
        if self._model is not None:
            self._model.clear_measurement()

    def _on_reset_calibration_clicked(self) -> None:
        if self._model is not None:
            self._model.clear_calibration()

    def _refresh_all(self) -> None:
        self._refresh_calib_status()
        self._refresh_measurement()
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
        self._btn_calibrate_points.setEnabled(self._has_image)
        self._btn_import_ratio.setEnabled(True)
        self._btn_export_ratio.setEnabled(scale_available)
        self._btn_apply_ratio.setEnabled(True)
        self._btn_clear_measurement.setEnabled(scale_available)
        self._btn_reset_calibration.setEnabled(scale_available)
