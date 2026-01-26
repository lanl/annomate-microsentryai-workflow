"""
Custom UI Widgets for AnnoMate.

This module provides specialized Qt widgets used throughout the application,
including a collapsible splitter handle and a table widget with cyclic navigation.
"""

from PyQt5.QtWidgets import QSplitter, QSplitterHandle, QTableWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent, QKeyEvent


class SidebarHandle(QSplitterHandle):
    """
    A Custom Splitter Handle that supports hover effects and double-click toggling.
    """

    def __init__(self, orientation: Qt.Orientation, parent: QSplitter):
        """
        Initialize the handle and enable hover attributes.

        Args:
            orientation (Qt.Orientation): The orientation of the splitter.
            parent (QSplitter): The parent splitter widget.
        """
        super().__init__(orientation, parent)
        # Enable hover attribute so that QSS :hover selectors work correctly
        self.setAttribute(Qt.WA_Hover, True)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """
        Toggles the visibility of the sidebar (second widget) on double-click.
        """
        splitter = self.splitter()
        sizes = splitter.sizes()

        # Ensure we have at least two widgets (Main Canvas, Sidebar)
        if len(sizes) < 2:
            return

        main_width, side_width = sizes[0], sizes[1]

        if side_width > 0:
            # COLLAPSE: Save current width and snap to 0
            splitter._last_side_width = side_width
            splitter.setSizes([main_width + side_width, 0])
        else:
            # EXPAND: Restore last known width or default to 400
            last_width = getattr(splitter, '_last_side_width', 400)
            if last_width == 0:
                last_width = 400

            # Calculate new sizes ensuring we don't exceed total width
            total_width = main_width + side_width
            splitter.setSizes([total_width - last_width, last_width])


class CustomSplitter(QSplitter):
    """
    A QSplitter subclass that utilizes the SidebarHandle for enhanced interaction.
    """

    def createHandle(self) -> QSplitterHandle:
        """Overrides the default handle creation to use SidebarHandle."""
        return SidebarHandle(self.orientation(), self)


class WrappingTableWidget(QTableWidget):
    """
    A QTableWidget that implements cyclic selection navigation.

    Pressing 'Down' on the last row wraps to the first row.
    Pressing 'Up' on the first row wraps to the last row.
    """

    def keyPressEvent(self, event: QKeyEvent):
        """
        Handles key press events to enforce cyclic navigation.
        """
        if event.key() == Qt.Key_Down:
            # If on the last row, jump to top
            if self.rowCount() > 0 and self.currentRow() == self.rowCount() - 1:
                self.selectRow(0)
                return

        elif event.key() == Qt.Key_Up:
            # If on the first row, jump to bottom
            if self.rowCount() > 0 and self.currentRow() == 0:
                self.selectRow(self.rowCount() - 1)
                return

        # Default behavior for other keys
        super().keyPressEvent(event)