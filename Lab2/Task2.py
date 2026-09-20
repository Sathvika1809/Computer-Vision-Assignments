import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QComboBox, 
    QMessageBox, QDialog, QSpinBox, QFormLayout, QSizePolicy
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

PAPER_SIZES = {
    "A4 (595 x 842)": (595, 842),
    "US Letter (612 x 792)": (612, 792),
    "US Legal (612 x 1008)": (612, 1008),
    "A3 (842 x 1191)": (842, 1191),
    "Square (700 x 700)": (700, 700),
    "Custom": None
}


class ClickableImageLabel(QLabel):
    """A QLabel that fits images strictly inside window bounds and tracks points."""

    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.points = []
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def mousePressEvent(self, event):
        if self.pixmap() is None or self.parent_window.original_bgr is None or len(self.points) >= 4:
            return

        # Fetch actual displayed pixmap dimensions inside label
        pixmap = self.pixmap()
        pw, ph = pixmap.width(), pixmap.height()
        lw, lh = self.width(), self.height()

        # Calculate offsets caused by Qt.AlignCenter
        offset_x = (lw - pw) / 2.0
        offset_y = (lh - ph) / 2.0

        pos = event.position()
        click_x = pos.x() - offset_x
        click_y = pos.y() - offset_y

        # Ignore clicks outside the actual image pixmap bounds
        if click_x < 0 or click_x >= pw or click_y < 0 or click_y >= ph:
            return

        # Map display click coordinates to real image pixel dimensions
        img_h, img_w = self.parent_window.original_bgr.shape[:2]
        real_x = int(click_x * (img_w / pw))
        real_y = int(click_y * (img_h / ph))

        self.points.append((real_x, real_y))
        self.parent_window.redraw_with_points()

    def reset_points(self):
        self.points = []

    def set_points(self, points):
        self.points = points
        self.parent_window.redraw_with_points()


class WarpedResultDialog(QDialog):
    """Displays the warped output properly fitted inside a PySide6 window."""

    def __init__(self, warped_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Warped Result")
        self.warped_bgr = warped_bgr
        self.resize(700, 700)

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
        self.display_buffer = np.ascontiguousarray(rgb)
        h, w, ch = self.display_buffer.shape
        qimg = QImage(self.display_buffer.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale output result to fit view
        scaled_pixmap = pixmap.scaled(650, 650, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.result_label.setPixmap(scaled_pixmap)

        save_button = QPushButton("Save As...")
        save_button.clicked.connect(self.save_image)

        layout = QVBoxLayout()
        layout.addWidget(self.result_label, 1)
        layout.addWidget(save_button, 0)
        self.setLayout(layout)

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Warped Image", "", "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if file_path:
            cv2.imwrite(file_path, self.warped_bgr)


class HomographyTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lab 2 - Automatic Scan-to-Paper-Size Tool")
        self.resize(1000, 750)

        self.original_bgr = None
        self.display_buffer = None

        # --- Image display area ---
        self.image_label = ClickableImageLabel(self)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b;")

        # --- Controls ---
        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)

        self.auto_detect_button = QPushButton("Auto-Detect Corners")
        self.auto_detect_button.clicked.connect(self.auto_detect_corners)

        self.reset_button = QPushButton("Reset Points")
        self.reset_button.clicked.connect(self.reset_points)

        self.paper_dropdown = QComboBox()
        self.paper_dropdown.addItems(PAPER_SIZES.keys())
        self.paper_dropdown.currentIndexChanged.connect(self.toggle_custom_size_inputs)

        # Custom dimensions inputs
        self.custom_width_box = QSpinBox()
        self.custom_width_box.setRange(100, 5000)
        self.custom_width_box.setValue(800)

        self.custom_height_box = QSpinBox()
        self.custom_height_box.setRange(100, 5000)
        self.custom_height_box.setValue(1000)

        self.custom_widget = QWidget()
        custom_layout = QFormLayout(self.custom_widget)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_layout.addRow("W (px):", self.custom_width_box)
        custom_layout.addRow("H (px):", self.custom_height_box)
        self.custom_widget.setVisible(False)

        self.warp_button = QPushButton("Compute & Warp")
        self.warp_button.clicked.connect(self.compute_and_warp)

        self.status_label = QLabel("Open an image, then auto-detect corners or click TL, TR, BR, BL manually.")

        controls_row = QHBoxLayout()
        controls_row.addWidget(self.open_button)
        controls_row.addWidget(self.auto_detect_button)
        controls_row.addWidget(self.reset_button)
        controls_row.addWidget(self.paper_dropdown)
        controls_row.addWidget(self.custom_widget)
        controls_row.addWidget(self.warp_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label, 1)  # Stretch image container to fill window
        main_layout.addLayout(controls_row, 0)
        main_layout.addWidget(self.status_label, 0)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def resizeEvent(self, event):
        """Redraw image with correct scale whenever main window is resized."""
        super().resizeEvent(event)
        if self.original_bgr is not None:
            if len(self.image_label.points) > 0:
                self.redraw_with_points()
            else:
                self.render_image(self.original_bgr)

    def toggle_custom_size_inputs(self):
        is_custom = self.paper_dropdown.currentText() == "Custom"
        self.custom_widget.setVisible(is_custom)

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
        self.status_label.setText("Image loaded. Click 'Auto-Detect Corners' or manual 4 points.")

    def reset_points(self):
        self.image_label.reset_points()
        if self.original_bgr is not None:
            self.render_image(self.original_bgr)
        self.status_label.setText("Points cleared. Click 4 corners again.")

    def redraw_with_points(self):
        if self.original_bgr is None:
            return

        preview = self.original_bgr.copy()
        labels = ["1: TL", "2: TR", "3: BR", "4: BL"]
        img_h, img_w = preview.shape[:2]

        # Dynamic marker size based on resolution
        radius = max(6, int(min(img_w, img_h) * 0.015))
        font_scale = max(0.6, min(img_w, img_h) * 0.0012)
        thickness = max(2, int(radius * 0.3))

        for i, (x, y) in enumerate(self.image_label.points):
            cv2.circle(preview, (int(x), int(y)), radius, (0, 0, 255), -1)
            cv2.putText(
                preview, labels[i] if i < 4 else str(i + 1), 
                (int(x) + radius + 2, int(y) - radius - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness
            )

        self.render_image(preview, keep_points=True)
        self.status_label.setText(f"{len(self.image_label.points)}/4 points selected.")

    def auto_detect_corners(self):
        if self.original_bgr is None:
            QMessageBox.warning(self, "Error", "Load an image first.")
            return

        gray = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Downsample image for edge detection stability across multi-MP images
        proc_scale = 800.0 / max(h, w)
        proc_w = int(w * proc_scale)
        proc_h = int(h * proc_scale)
        proc_gray = cv2.resize(gray, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        blurred = cv2.GaussianBlur(proc_gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to bridge gaps in document border
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_pts = None

        if contours:
            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for c in contours:
                area = cv2.contourArea(c)
                if area < (proc_w * proc_h * 0.15):  # Filter out small contours
                    continue

                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                if len(approx) == 4:
                    detected_pts = approx.reshape(4, 2) / proc_scale
                    break

            # Secondary Strategy: Minimum Area Rotated Rectangle
            if detected_pts is None and len(contours) > 0:
                largest_c = contours[0]
                if cv2.contourArea(largest_c) > (proc_w * proc_h * 0.1):
                    rect = cv2.minAreaRect(largest_c)
                    box = cv2.boxPoints(rect)
                    detected_pts = box / proc_scale

        # Fallback to outer edges of entire frame if no contour was identified
        if detected_pts is None:
            detected_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            self.status_label.setText("No inner document border found. Snapped to full image frame.")
        else:
            self.status_label.setText("Document corners auto-detected successfully.")

        ordered_corners = self.order_points(detected_pts)
        self.image_label.set_points([(int(p[0]), int(p[1])) for p in ordered_corners])

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def compute_and_warp(self):
        if self.original_bgr is None:
            QMessageBox.warning(self, "Error", "Load an image first.")
            return
        if len(self.image_label.points) != 4:
            QMessageBox.warning(self, "Error", "Select or auto-detect exactly 4 points first.")
            return

        selected_preset = self.paper_dropdown.currentText()
        if selected_preset == "Custom":
            target_w = self.custom_width_box.value()
            target_h = self.custom_height_box.value()
        else:
            target_w, target_h = PAPER_SIZES[selected_preset]

        src_points = np.float32(self.image_label.points)
        dst_points = np.float32([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1],
        ])

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

        # Scale pixmap smoothly to fit inside available view while maintaining aspect ratio
        target_size = self.image_label.size()
        scaled_pixmap = pixmap.scaled(
            target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

        if not keep_points:
            self.image_label.reset_points()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HomographyTool()
    window.show()
    sys.exit(app.exec())