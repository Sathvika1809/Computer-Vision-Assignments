#Introducing GUIs: PySide6
# cv2.imshow() - basic and not interactive

import sys
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QScrollArea,
    QFileDialog,
)


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIPI Interactive Image Viewer (Scale / Rotate / Flip)")
        self.resize(1000, 750)

        self.image_bgr = None  # Original master copy, never modified in-place

        self._init_ui()
        self._update_controls_state()

    def _init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Control Panel Layout
        ctrl_layout = QHBoxLayout()

        # Open button
        self.btn_open = QPushButton("Open Image")
        self.btn_open.clicked.connect(self.open_image)
        ctrl_layout.addWidget(self.btn_open)

        # Scale Slider (10% to 200%)
        ctrl_layout.addWidget(QLabel("Scale:"))
        self.slider_scale = QSlider(Qt.Horizontal)
        self.slider_scale.setRange(10, 200)
        self.slider_scale.setValue(100)
        self.slider_scale.valueChanged.connect(self.apply_transforms)
        ctrl_layout.addWidget(self.slider_scale)
        self.label_scale_val = QLabel("100%")
        ctrl_layout.addWidget(self.label_scale_val)

        # Rotate Slider (0 to 360) + Angle Label
        ctrl_layout.addWidget(QLabel("Rotate:"))
        self.slider_rotate = QSlider(Qt.Horizontal)
        self.slider_rotate.setRange(0, 360)
        self.slider_rotate.setValue(0)
        self.slider_rotate.valueChanged.connect(self.apply_transforms)
        ctrl_layout.addWidget(self.slider_rotate)
        self.label_rotate_val = QLabel("0°")
        ctrl_layout.addWidget(self.label_rotate_val)

        # Flip Horizontal / Vertical toggle buttons
        self.btn_flip_h = QPushButton("Flip H")
        self.btn_flip_h.setCheckable(True)
        self.btn_flip_h.clicked.connect(self.apply_transforms)
        ctrl_layout.addWidget(self.btn_flip_h)

        self.btn_flip_v = QPushButton("Flip V")
        self.btn_flip_v.setCheckable(True)
        self.btn_flip_v.clicked.connect(self.apply_transforms)
        ctrl_layout.addWidget(self.btn_flip_v)

        main_layout.addLayout(ctrl_layout)

        # Scroll Area for True-Size Image Display
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)

        main_layout.addWidget(self.scroll_area)
        self.setCentralWidget(central_widget)

    def _update_controls_state(self):
        has_img = self.image_bgr is not None
        self.slider_scale.setEnabled(has_img)
        self.slider_rotate.setEnabled(has_img)
        self.btn_flip_h.setEnabled(has_img)
        self.btn_flip_v.setEnabled(has_img)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open SIPI Image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )
        if file_path:
            loaded = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if loaded is not None:
                self.image_bgr = loaded
                self._update_controls_state()
                self.apply_transforms()

    def apply_transforms(self):
        if self.image_bgr is None:
            return

        # Read controls
        scale_pct = self.slider_scale.value()
        angle_deg = self.slider_rotate.value()
        flip_h = self.btn_flip_h.isChecked()
        flip_v = self.btn_flip_v.isChecked()

        # Update labels
        self.label_scale_val.setText(f"{scale_pct}%")
        self.label_rotate_val.setText(f"{angle_deg}°")

        # 1. Always transform a fresh copy of self.image_bgr
        working_img = self.image_bgr.copy()

        # 2. Scale
        s = scale_pct / 100.0
        if s != 1.0:
            working_img = cv2.resize(working_img, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

        # 3. Unclipped Rotation
        h, w = working_img.shape[:2]
        if angle_deg != 0:
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            # Calculate unclipped bounding box
            cos_val = np.abs(M[0, 0])
            sin_val = np.abs(M[0, 1])
            new_w = int((h * sin_val) + (w * cos_val))
            new_h = int((h * cos_val) + (w * sin_val))
            # Shift translation to account for new center
            M[0, 2] += (new_w / 2.0) - center[0]
            M[1, 2] += (new_h / 2.0) - center[1]
            working_img = cv2.warpAffine(
                working_img, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )

        # 4. Flips
        if flip_h and flip_v:
            working_img = cv2.flip(working_img, -1)
        elif flip_h:
            working_img = cv2.flip(working_img, 1)
        elif flip_v:
            working_img = cv2.flip(working_img, 0)

        # 5. Convert OpenCV BGR to RGB / QPixmap for true-size QScrollArea rendering
        rgb_img = cv2.cvtColor(working_img, cv2.COLOR_BGR2RGB)
        rh, rw, ch = rgb_img.shape
        bytes_per_line = ch * rw
        q_img = QImage(rgb_img.data, rw, rh, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Display at true size and force scrollbar recalculation
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageViewer()
    window.show()
    sys.exit(app.exec())