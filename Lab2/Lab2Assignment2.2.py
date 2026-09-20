import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QComboBox, QMessageBox, QDialog
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

# Paper sizes in pixels using standard 72-DPI convention (portrait orientation)
PAPER_SIZES = {
    "A4 (595 x 842)": (595, 842),
    "US Letter (612 x 792)": (612, 792),
    "US Legal (612 x 1008)": (612, 1008),
    "A3 (842 x 1191)": (842, 1191),
    "Square (700 x 700)": (700, 700),
}


class ClickableImageLabel(QLabel):
    """A QLabel that records up to 4 click points in image-pixel coordinates."""

    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.points = []  # list of (x, y) tuples, up to 4

    def mousePressEvent(self, event):
        if self.pixmap() is None or len(self.points) >= 4:
            return
        pos = event.position().toPoint()
        self.points.append((pos.x(), pos.y()))
        self.parent_window.redraw_with_points()

    def reset_points(self):
        self.points = []


class WarpedResultDialog(QDialog):
    """
    Displays the warped output inside a PySide6 window with a Save button.
    Avoids cv2.imshow() to prevent crashes in headless environments.
    """

    def __init__(self, warped_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Warped Result")
        self.warped_bgr = warped_bgr
        self.resize(700, 700)

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.result_label)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setMinimumSize(600, 600)

        # Convert BGR (OpenCV) to RGB (PySide6)
        rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
        self.display_buffer = np.ascontiguousarray(rgb)  # Keep buffer alive in memory
        h, w, ch = self.display_buffer.shape
        qimg = QImage(self.display_buffer.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        self.result_label.setPixmap(pixmap)
        self.result_label.resize(pixmap.size())

        save_button = QPushButton("Save As...")
        save_button.clicked.connect(self.save_image)

        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)
        layout.addWidget(save_button)
        self.setLayout(layout)

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Warped Image", "",
            "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if file_path:
            cv2.imwrite(file_path, self.warped_bgr)


class HomographyTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lab 2 - Manual Scan-to-Paper-Size Tool")
        self.resize(900, 750)

        self.original_bgr = None
        self.display_buffer = None  # Keeps QImage's backing memory alive

        # --- Image display area ---
        self.image_label = ClickableImageLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setMinimumSize(700, 500)

        # --- Controls ---
        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)

        self.reset_button = QPushButton("Reset Points")
        self.reset_button.clicked.connect(self.reset_points)

        self.paper_dropdown = QComboBox()
        self.paper_dropdown.addItems(PAPER_SIZES.keys())

        self.warp_button = QPushButton("Compute & Warp")
        self.warp_button.clicked.connect(self.compute_and_warp)

        self.status_label = QLabel("Open an image, then click 4 corners: TL, TR, BR, BL.")

        controls_row = QHBoxLayout()
        controls_row.addWidget(self.open_button)
        controls_row.addWidget(self.reset_button)
        controls_row.addWidget(self.paper_dropdown)
        controls_row.addWidget(self.warp_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.scroll_area)
        main_layout.addLayout(controls_row)
        main_layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "All Files (*);;Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.gif)"
        )
        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            QMessageBox.warning(self, "Error", "Failed to load image.")
            return

        self.original_bgr = image
        self.image_label.reset_points()
        self.render_image(self.original_bgr)
        self.status_label.setText("Click 4 corners, in order: top-left, top-right, bottom-right, bottom-left.")

    def reset_points(self):
        self.image_label.reset_points()
        if self.original_bgr is not None:
            self.render_image(self.original_bgr)
        self.status_label.setText("Points cleared. Click 4 corners again.")

    def redraw_with_points(self):
        """Draw red circular markers and labels for clicked coordinates."""
        if self.original_bgr is None:
            return

        preview = self.original_bgr.copy()
        for i, (x, y) in enumerate(self.image_label.points):
            cv2.circle(preview, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(preview, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self.render_image(preview, keep_points=True)
        self.status_label.setText(f"{len(self.image_label.points)}/4 points selected.")

    def compute_and_warp(self):
        """Calculates perspective transformation matrix and warps region to standard target dimensions."""
        if self.original_bgr is None:
            QMessageBox.warning(self, "Error", "Load an image first.")
            return
        if len(self.image_label.points) != 4:
            QMessageBox.warning(self, "Error", "Select exactly 4 points first.")
            return

        target_w, target_h = PAPER_SIZES[self.paper_dropdown.currentText()]

        # Source coordinates selected by user
        src_points = np.float32(self.image_label.points)
        
        # Target destination corners mapping to standard page orientation
        dst_points = np.float32([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1],
        ])

        # Compute Homography Matrix and apply transformation
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(self.original_bgr, M, (target_w, target_h))

        dialog = WarpedResultDialog(warped, self)
        dialog.exec()

    def render_image(self, cv_image, keep_points=False):
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.display_buffer = np.ascontiguousarray(rgb)
        h, w, ch = self.display_buffer.shape
        bytes_per_line = ch * w
        
        qimg = QImage(self.display_buffer.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())

        if not keep_points:
            self.image_label.reset_points()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HomographyTool()
    window.show()
    sys.exit(app.exec())