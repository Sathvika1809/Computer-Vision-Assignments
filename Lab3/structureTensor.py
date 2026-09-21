import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QSpinBox, QMessageBox
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


class ClickableImageLabel(QLabel):
    """A QLabel that reports click coordinates (in image-pixel space) to its parent."""

    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window

    def mousePressEvent(self, event):
        if self.pixmap() is None:
            return
        pos = event.position().toPoint()
        self.parent_window.handle_click(pos.x(), pos.y())


class StructureTensorViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lab 3 - Structure Tensor Ellipse Viewer")
        self.resize(900, 750)

        self.original_bgr = None
        self.gray = None
        self.Ix2 = None   # precomputed Ix^2, once per loaded image
        self.Iy2 = None   # precomputed Iy^2
        self.Ixy = None   # precomputed Ix*Iy
        self.click_points = []
        self.display_buffer = None  # keeps QImage's backing memory alive

        # --- Image display (true size, inside a scroll area) ---
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

        self.clear_button = QPushButton("Clear Points")
        self.clear_button.clicked.connect(self.clear_points)

        self.window_size_box = QSpinBox()
        self.window_size_box.setRange(5, 101)
        self.window_size_box.setSingleStep(2)
        self.window_size_box.setValue(21)

        self.scale_box = QSpinBox()
        self.scale_box.setRange(1, 2000)
        self.scale_box.setValue(300)

        self.max_axis_box = QSpinBox()
        self.max_axis_box.setRange(5, 400)
        self.max_axis_box.setValue(100)

        controls_row = QHBoxLayout()
        controls_row.addWidget(self.open_button)
        controls_row.addWidget(QLabel("Window size (px):"))
        controls_row.addWidget(self.window_size_box)
        controls_row.addWidget(QLabel("Ellipse scale:"))
        controls_row.addWidget(self.scale_box)
        controls_row.addWidget(QLabel("Max axis (px):"))
        controls_row.addWidget(self.max_axis_box)
        controls_row.addWidget(self.clear_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.scroll_area)
        main_layout.addLayout(controls_row)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "All Files (*);;Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.ppm)"
        )
        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            QMessageBox.warning(self, "Error", "Failed to load image.")
            return

        self.original_bgr = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Precompute gradients and gradient-product images ONCE per image,
        # so each click only has to average a small window, not recompute Sobel.
        Ix = cv2.Sobel(self.gray, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(self.gray, cv2.CV_32F, 0, 1, ksize=3)
        self.Ix2 = Ix * Ix
        self.Iy2 = Iy * Iy
        self.Ixy = Ix * Iy

        self.click_points = []
        self.render_image(self.original_bgr)

    def clear_points(self):
        self.click_points = []
        if self.original_bgr is not None:
            self.render_image(self.original_bgr)

    def handle_click(self, x, y):
        if self.original_bgr is None:
            return
        self.click_points.append((x, y))
        self.redraw_with_ellipses()

    def structure_tensor_at(self, x, y, half_win):
        """Average the precomputed gradient-product images over a window
        centered at (x, y), then eigen-decompose the resulting 2x2 matrix.

        We use the window MEAN (not the raw sum) so eigenvalue magnitudes
        stay comparable across different window sizes."""
        h, w = self.gray.shape
        x0, x1 = max(0, x - half_win), min(w, x + half_win + 1)
        y0, y1 = max(0, y - half_win), min(h, y + half_win + 1)

        Sxx = float(np.mean(self.Ix2[y0:y1, x0:x1]))
        Syy = float(np.mean(self.Iy2[y0:y1, x0:x1]))
        Sxy = float(np.mean(self.Ixy[y0:y1, x0:x1]))

        M = np.array([[Sxx, Sxy], [Sxy, Syy]], dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(M)  # ascending order

        lam2, lam1 = eigenvalues[0], eigenvalues[1]     # lam1 = larger
        v2, v1 = eigenvectors[:, 0], eigenvectors[:, 1]  # matching eigenvectors

        return lam1, lam2, v1, v2

    def redraw_with_ellipses(self):
        preview = self.original_bgr.copy()
        half_win = self.window_size_box.value() // 2
        scale = self.scale_box.value()
        max_axis = self.max_axis_box.value()
        epsilon = 1.0  # prevents division blow-up on near-zero (flat-region) eigenvalues

        for (x, y) in self.click_points:
            lam1, lam2, v1, v2 = self.structure_tensor_at(x, y, half_win)

            lam1 = max(lam1, 0.0)  # guard against tiny negative floating-point noise
            lam2 = max(lam2, 0.0)

            # Axis length is INVERSELY proportional to sqrt(eigenvalue):
            # a large eigenvalue means the gradient is well-determined in that
            # direction, so the point is tightly localized there (SMALL axis).
            # A near-zero eigenvalue (flat region) means nothing constrains
            # the point in that direction, so the axis should be LARGE.
            axis1 = scale / np.sqrt(lam1 + epsilon)  # goes with v1 (larger eigenvalue -> smaller axis)
            axis2 = scale / np.sqrt(lam2 + epsilon)  # goes with v2 (smaller eigenvalue -> larger axis)

            axis1 = int(np.clip(axis1, 2, max_axis))
            axis2 = int(np.clip(axis2, 2, max_axis))
            angle_deg = float(np.degrees(np.arctan2(v1[1], v1[0])))

            cv2.ellipse(preview, (x, y), (axis1, axis2), angle_deg, 0, 360, (0, 255, 255), 2)
            cv2.circle(preview, (x, y), 3, (0, 0, 255), -1)

        self.render_image(preview, keep_points=True)

    def render_image(self, cv_image, keep_points=False):
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.display_buffer = np.ascontiguousarray(rgb)
        h, w, ch = self.display_buffer.shape
        qimg = QImage(self.display_buffer.data, w, h, ch * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())

        if not keep_points:
            self.click_points = []


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StructureTensorViewer()
    window.show()
    sys.exit(app.exec())