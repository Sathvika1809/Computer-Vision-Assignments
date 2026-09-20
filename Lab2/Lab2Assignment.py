# Build a crack detector using only clasical image processing
# no machine learning, no trained classifiers
# Two folders - Positive/ Negative/
# Key Features
# Thresholding (fixed, Otsu, or adaptive - try more than one)
# Morphological Operations(erosion/dilation/opening/closing) to clean up thresholded mask
# Some rule you design yourself to turn cleaned mask into binary decision 
# "crack" or "no crack" for given image
# idea 1 - proportion of foreground pixels in cleaned mask
# idea 2 - Whether any contour is long and thin(high aspect ratio / low area-to-perimeter ratio),
# which is characteristic of a crack shape, vs blob like noise
# Edge density (fraction of piels marked as edges by Canny after cleanup)
import os
import random
import cv2

POSITIVE_DIR = "surface-crack-detection/Positive"
NEGATIVE_DIR = "surface-crack-detection/Negative"
SAMPLE_SIZE_PER_CLASS = 300  # start small while tuning, increase later


def is_crack(image_path, **params) -> bool:
    thresh_type = params.get("thresh_type", "adaptive")
    fixed_thresh = params.get("fixed_thresh", 110)
    block_size = params.get("block_size", 17)
    c_val = params.get("c_val", 5)
    kernel_size = params.get("kernel_size", 3)
    min_area = params.get("min_area", 60)
    min_aspect_ratio = params.get("min_aspect_ratio", 3.2)
    # Load image and convert to grayscale
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    # Pre-smooth to remove grain/rough textures
    blurred = cv2.GaussianBlur(img, (5,5), 0)

    # 2. Thresholding Strategy
    if thresh_type == "adaptive":
        # Cracks are darker than local background -> THRESH_BINARY_INV
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c_val
        )
    elif thresh_type == "otsu":
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:  # Fixed
        _, binary = cv2.threshold(
            blurred, fixed_thresh, 255, cv2.THRESH_BINARY_INV
        )

    # 3. Morphological Filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Close to bridge small gaps along the crack path
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # Open to clear away remaining single-pixel speckle noise
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # 4. Decision Rule: Geometric Contour Analysis
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            # Fit a minimum-area rotated bounding box
            rect = cv2.minAreaRect(cnt)
            (w, h) = rect[1]
            if w == 0 or h == 0:
                continue

            # Compute length-to-width ratio (aspect ratio)
            aspect_ratio = max(w, h) / min(w, h)

            # High aspect ratio indicates a long, thin shape (crack) vs a round blob (surface pit)
            if aspect_ratio >= min_aspect_ratio:
                return True

    return False


def evaluate(positive_dir, negative_dir, sample_size, **params):
    random.seed(42)
    positive_files = random.sample(os.listdir(positive_dir), sample_size)
    negative_files = random.sample(os.listdir(negative_dir), sample_size)

    tp = fp = tn = fn = 0

    for fname in positive_files:
        predicted = is_crack(os.path.join(positive_dir, fname), **params)
        if predicted:
            tp += 1
        else:
            fn += 1

    for fname in negative_files:
        predicted = is_crack(os.path.join(negative_dir, fname), **params)
        if predicted:
            fp += 1
        else:
            tn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Accuracy={accuracy:.3f}  Precision={precision:.3f}  Recall={recall:.3f}")
    return tp, fp, tn, fn


if __name__ == "__main__":
    evaluate(POSITIVE_DIR, NEGATIVE_DIR, SAMPLE_SIZE_PER_CLASS)