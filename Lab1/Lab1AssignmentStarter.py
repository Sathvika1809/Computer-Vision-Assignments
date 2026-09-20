import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_sipi_image(image_path, title_prefix):
    # 1. Load Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 2. Inspect shape & channels
    print(f"\n--- {title_prefix} ---")
    print(f"Shape: {img_bgr.shape} (Height, Width, Channels)")
    print(f"Data type: {img_bgr.dtype}")

    # 3. Grayscale conversion
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr.copy()

    # 4. Scaled & Rotated (Clipped vs Unclipped)
    # Scale (0.7x)
    scaled_img = cv2.resize(img_bgr, (0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
    
    # Rotation parameters (45 degrees)
    h, w = img_bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    angle = 45
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Clipped rotation
    rot_clipped = cv2.warpAffine(img_bgr, M, (w, h))

    # Unclipped rotation
    cos_val = np.abs(M[0, 0])
    sin_val = np.abs(M[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))
    M_unclipped = M.copy()
    M_unclipped[0, 2] += (new_w / 2.0) - center[0]
    M_unclipped[1, 2] += (new_h / 2.0) - center[1]
    rot_unclipped = cv2.warpAffine(img_bgr, M_unclipped, (new_w, new_h))

    # 5. RGB / HSV spaces
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 6. Manual Mirror (Horizontal flip without cv2.flip)
    manual_mirror = img_bgr[:, ::-1, :]

    # 7. Drawing Demo
    drawing_demo = img_bgr.copy()
    cv2.rectangle(drawing_demo, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 0), 3)
    cv2.circle(drawing_demo, (int(center[0]), int(center[1])), min(w, h) // 6, (0, 0, 255), -1)

    # Visualization Gallery
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Pipeline Demonstration: {title_prefix}", fontsize=16)

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original (RGB)")

    axes[0, 1].imshow(img_gray, cmap='gray')
    axes[0, 1].set_title("Grayscale")

    axes[0, 2].imshow(cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Scaled (0.7x)")

    axes[0, 3].imshow(cv2.cvtColor(rot_clipped, cv2.COLOR_BGR2RGB))
    axes[0, 3].set_title("Rotated (Clipped)")

    axes[1, 0].imshow(cv2.cvtColor(rot_unclipped, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Rotated (Unclipped)")

    axes[1, 1].imshow(img_hsv)
    axes[1, 1].set_title("HSV Color Space")

    axes[1, 2].imshow(cv2.cvtColor(manual_mirror, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Manual Mirror")

    axes[1, 3].imshow(cv2.cvtColor(drawing_demo, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title("Drawing Demo")

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Updated relative paths based on your folder structure
texture_image_path = "sipi-dataset/textures/1.1.02.tiff"
misc_image_path = "sipi-dataset/misc/misc/4.2.03.tiff"

# Execute Pipeline on Texture Image
process_sipi_image(texture_image_path, "Texture Volume Image")

# Execute Pipeline on Miscellaneous/Natural Image
process_sipi_image(misc_image_path, "Miscellaneous Volume Image")