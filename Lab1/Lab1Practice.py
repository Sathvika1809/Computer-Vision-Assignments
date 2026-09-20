import cv2

img = cv2.imread("sipi-dataset/misc/misc/4.2.07.tiff")  # returns a NumPy array (BGR order)

if img is None:
    raise FileNotFoundError("Could not read the image — check the path.")

# Displaying images in different window modes
cv2.namedWindow("AutoSize Window", cv2.WINDOW_AUTOSIZE)
cv2.imshow("AutoSize Window", img)

# Resizable window: user can drag to resize: image is scaled to fit
cv2.namedWindow("Resizable Window", cv2.WINDOW_NORMAL)
cv2.imshow("Resizable Window",img)

# Fixed-size Window: Force a specific window size regardless of image size
cv2.namedWindow("Fixed Size Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Fixed Size Window", 400, 300)
cv2.imshow("Fixed Size Window",img)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("output.png", img)     # OpenCV infers format from the extension
cv2.imwrite("output.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])

# GrayScale Conversion, Channels, and Dimensions
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #BGR2GRAY -> bgr to gray
# returns a numpy array
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Inspecting image properties
print("Image dtype:", img.dtype)                 # usually uint8
print("Shape (H, W, C):", img.shape)              # color image -> 3 dims (height,width,color)
print("Shape (H, W):", gray.shape)                # grayscale image -> 2 dims (height,width)
print("Height:", img.shape[0], "Width:", img.shape[1])

if img.ndim == 3:
    print("Number of channels:", img.shape[2])
else:
    print("Number of channels: 1 (grayscale)")

print("Total pixels:", img.shape[0] * img.shape[1])
print("Total size in memory (bytes):", img.size)

# Creating a Empty Image
import numpy as np

# Black canvas: 512 rows, 512 cols, 3 color channels, 8-bit per channel
blank_color = np.zeros((512, 512, 3), dtype=np.uint8)

# Black grayscale canvas
blank_gray = np.zeros((512, 512), dtype=np.uint8)

# White canvas
blank_white = np.full((512, 512, 3), 255, dtype=np.uint8)

#Scaling and Rotating Images
h, w = img.shape[:2] # extracts h and w in pixels from an image array in Python

# Scale by a fixed factor
scaled_up = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR) # Linear for enlarging
scaled_down = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) # Area for diminishing

# Scale to an exact target size
scaled_fixed = cv2.resize(img, (300, 200), interpolation=cv2.INTER_LINEAR)

# Rotation about image center
angle = 45  # hardcoded rotation angle, in degrees (counter-clockwise)

(h, w) = img.shape[:2]
(cX, cY) = (w // 2, h // 2)   # rotate about the image center

M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)  # 1.0 = no additional scaling

# ---- Clipped rotation: same output dimensions as input ----
rotated_clipped = cv2.warpAffine(img, M, (w, h))

# ---- Unclipped rotation: expand canvas to fit the whole rotated image ----
cos = abs(M[0, 0])
sin = abs(M[0, 1])

new_w = int((h * sin) + (w * cos))
new_h = int((h * cos) + (w * sin))

# Shift the rotation matrix so the image is centered in the new, larger canvas
M[0, 2] += (new_w / 2) - cX
M[1, 2] += (new_h / 2) - cY

rotated_unclipped = cv2.warpAffine(img, M, (new_w, new_h))

cv2.imshow("Clipped Rotation", rotated_clipped)
cv2.imshow("Unclipped Rotation", rotated_unclipped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Visualizing RGB and HSV Channels(Matplotlib Gallery)

import matplotlib.pyplot as plt

img_bgr = cv2.imread("sipi-dataset/misc/misc/4.2.07.tiff")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

R, G, B = cv2.split(img_rgb)
H, S, V = cv2.split(img_hsv)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original (RGB)")

axes[0, 1].imshow(R, cmap="Reds")
axes[0, 1].set_title("Red Channel")

axes[0, 2].imshow(G, cmap="Greens")
axes[0, 2].set_title("Green Channel")

axes[0, 3].imshow(B, cmap="Blues")
axes[0, 3].set_title("Blue Channel")

axes[1, 0].imshow(img_hsv)
axes[1, 0].set_title("Original (as HSV array)")

axes[1, 1].imshow(H, cmap="hsv")
axes[1, 1].set_title("Hue")

axes[1, 2].imshow(S, cmap="gray")
axes[1, 2].set_title("Saturation")

axes[1, 3].imshow(V, cmap="gray")
axes[1, 3].set_title("Value")

for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
plt.show()

# Manual Pixel Access: Mirroring an Image
# In practice we use cv2.flip()


import numpy as np

img = cv2.imread("sipi-dataset/misc/misc/4.2.07.tiff")
h, w = img.shape[:2]

# ---- Horizontal mirror (flip left-right) ----
# if u want to change the org image then temp variable stuff
# We are writing everything new  zero_like means onto black board
mirrored_horizontal = np.zeros_like(img)
for i in range(h):
    for j in range(w):
        mirrored_horizontal[i, j] = img[i, w - 1 - j]

# ---- Vertical mirror (flip top-bottom) ----
mirrored_vertical = np.zeros_like(img)
for i in range(h):
    for j in range(w):
        mirrored_vertical[i, j] = img[h - 1 - i, j]

cv2.imshow("Original", img)
cv2.imshow("Mirrored Horizontal", mirrored_horizontal)
cv2.imshow("Mirrored Vertical", mirrored_vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sanity check against OpenCV's built-in flip:
assert np.array_equal(mirrored_horizontal, cv2.flip(img, 1))
assert np.array_equal(mirrored_vertical, cv2.flip(img, 0))

# Drawing Shapes and Text

import numpy as np

canvas = np.zeros((512, 512, 3), dtype=np.uint8)

# Line: start point, end point, color (BGR), thickness
cv2.line(canvas, (50, 50), (450, 50), (0, 255, 0), 3)

# Circle: center, radius, color, thickness (-1 = filled)
cv2.circle(canvas, (256, 256), 100, (255, 0, 0), 2)
cv2.circle(canvas, (256, 256), 20, (0, 0, 255), -1)

# Text: text, bottom-left origin, font, scale, color, thickness
cv2.putText(
    canvas,
    "OpenCV Lab 1",
    (100, 400),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2,
    cv2.LINE_AA,
)

cv2.imshow("Canvas", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
