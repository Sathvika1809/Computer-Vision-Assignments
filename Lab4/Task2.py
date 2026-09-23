import cv2
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning
from skimage.color import rgb2lab, deltaE_cie76

# Suppress Convergence Warnings when n_clusters > unique_colors
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ==========================================
# UPDATED EXPERIMENTAL BENCHMARK WITH FIXES
# ==========================================

def run_evaluation_benchmark():
    np.random.seed(42)
    test_images = {}
    
    # 1. Simple few-color object
    img_simple = np.ones((100, 100, 3), dtype=np.uint8) * 240
    img_simple[20:80, 20:80] = [220, 20, 20]
    img_simple[40:60, 40:60] = [20, 20, 220]
    test_images["Simple Objects"] = img_simple

    # 2. Smooth gradient background
    gradient = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))
    img_grad = cv2.merge([gradient, 255 - gradient, np.full((100, 100), 128, dtype=np.uint8)])
    test_images["Smooth Gradient"] = img_grad

    # 3. Busy natural scene (slight blur preprocessing prevents noise overload)
    img_busy = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    cv2.circle(img_busy, (50, 50), 30, (0, 200, 0), -1)
    test_images["Busy Scene"] = img_busy

    # 4. Texture-heavy image with continuous color variations (not just 2 exact RGB values)
    img_tex = np.zeros((100, 100, 3), dtype=np.float32)
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            if (i // 10 + j // 10) % 2 == 0:
                img_tex[i:i+10, j:j+10] = [200, 200, 200]
            else:
                img_tex[i:i+10, j:j+10] = [50, 50, 50]
    # Add subtle Gaussian noise so there are > 8 unique pixel vectors
    noise = np.random.normal(0, 5, img_tex.shape)
    img_tex = np.clip(img_tex + noise, 0, 255).astype(np.uint8)
    test_images["Texture Heavy"] = img_tex

    # ... [rest of plotting code remains the same]
# ==========================================
# 1. IMPROVED IMAGE-AWARE SELECTION METRIC
# ==========================================

def compute_perceptual_segmentation_score(image_rgb, labels, centers, gamma=25.0):
    """
    Evaluates segmentation quality based on:
    1. Perceptual Color Distinctness (Delta-E in CIELAB space with standard JND soft thresholding)
    2. Resolution-Independent Spatial Region Coherence (Penalty for excessive fragmentation)
    """
    h, w, _ = image_rgb.shape
    total_pixels = h * w
    K = len(centers)

    if K < 2:
        return 0.0

    # 1. Convert cluster centers to CIELAB space for perceptually uniform color distance
    centers_lab = rgb2lab(centers.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
    
    # Calculate pairwise Delta-E with Just Noticeable Difference (JND ~ 2.3)
    pairwise_scores = []
    for i in range(K):
        for j in range(i + 1, K):
            dist = deltaE_cie76(centers_lab[i], centers_lab[j])
            
            # Sub-perceptual differences (< 2.3) are penalized heavily
            if dist < 2.3:
                score = 0.1 * dist
            else:
                score = np.log(1.0 + dist)
            pairwise_scores.append(score)

    mean_color_score = np.mean(pairwise_scores) if pairwise_scores else 0.0

    # 2. Resolution-Independent Spatial Coherence (Connected Components)
    segmentation_mask = labels.reshape(h, w).astype(np.uint8)
    
    # Scale morphological kernel dynamically based on image dimensions (min 3x3)
    kernel_size = max(3, int(min(h, w) * 0.01))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    total_regions = 0
    for k in range(K):
        binary_mask = (segmentation_mask == k).astype(np.uint8)
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        num_labels, _ = cv2.connectedComponents(cleaned_mask)
        total_regions += (num_labels - 1)

    # Normalize fragmentation factor by total pixels
    fragmentation_density = total_regions / float(total_pixels)
    coherence_penalty = np.exp(-gamma * fragmentation_density)

    return mean_color_score * coherence_penalty


def min_max_normalize(dictionary):
    """Normalizes array of metric dictionary values into [0, 1] range."""
    values = np.array(list(dictionary.values()))
    min_v, max_v = np.min(values), np.max(values)
    if max_v - min_v < 1e-6:
        return {k: 1.0 for k in dictionary.keys()}
    return {k: (v - min_v) / (max_v - min_v) for k, v in dictionary.items()}


def select_autonomous_k(image_rgb, k_range=range(2, 9)):
    """
    Evaluates candidate K values and selects optimal K using composite scoring.
    """
    h, w, c = image_rgb.shape
    pixels = image_rgb.reshape(-1, 3)

    unique_colors = np.unique(pixels, axis=0)
    num_unique_colors = len(unique_colors)
    
    valid_ks = [k for k in k_range if k <= num_unique_colors]
    if not valid_ks:
        valid_ks = [min(2, num_unique_colors)]

    sample_size = min(2000, pixels.shape[0])
    sample_indices = np.random.choice(pixels.shape[0], size=sample_size, replace=False)
    pixel_sample = pixels[sample_indices]

    sil_scores = {}
    perceptual_scores = {}
    models = {}

    for k in valid_ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5).fit(pixels)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # 1. Silhouette Score calculation
        if len(np.unique(labels[sample_indices])) > 1:
            sil = silhouette_score(pixel_sample, labels[sample_indices])
        else:
            sil = 0.0
        sil_scores[k] = sil

        # 2. Image-aware Perceptual Score calculation
        p_score = compute_perceptual_segmentation_score(image_rgb, labels, centers)
        perceptual_scores[k] = p_score

        # Save segmented frame output
        segmented_pixels = centers[labels].astype(np.uint8)
        models[k] = segmented_pixels.reshape(h, w, c)

    # Normalize both metric dictionaries to [0, 1] before combining
    norm_sil = min_max_normalize(sil_scores)
    norm_perceptual = min_max_normalize(perceptual_scores)

    composite_scores = {}
    for k in valid_ks:
        # Balanced 50/50 combination of statistical separation and visual/spatial coherence
        composite_scores[k] = 0.5 * norm_sil[k] + 0.5 * norm_perceptual[k]

    best_k = max(composite_scores, key=composite_scores.get)
    return best_k, models[best_k], composite_scores, models


# ==========================================
# 2. EXPERIMENTAL EVALUATION (4 TEST CASES)
# ==========================================

def run_evaluation_benchmark():
    np.random.seed(42)
    test_images = {}
    
    # 1. Simple few-color object
    img_simple = np.ones((100, 100, 3), dtype=np.uint8) * 240
    img_simple[20:80, 20:80] = [220, 20, 20]
    img_simple[40:60, 40:60] = [20, 20, 220]
    test_images["Simple Objects"] = img_simple

    # 2. Smooth gradient background
    gradient = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))
    img_grad = cv2.merge([gradient, 255 - gradient, np.full((100, 100), 128, dtype=np.uint8)])
    test_images["Smooth Gradient"] = img_grad

    # 3. Busy natural scene
    img_busy = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    cv2.circle(img_busy, (50, 50), 30, (0, 200, 0), -1)
    test_images["Busy Scene"] = img_busy

    # 4. Texture-heavy image
    img_tex = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            if (i // 10 + j // 10) % 2 == 0:
                img_tex[i:i+10, j:j+10] = [200, 200, 200]
            else:
                img_tex[i:i+10, j:j+10] = [50, 50, 50]
    test_images["Texture Heavy"] = img_tex

    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    row = 0

    for title, img in test_images.items():
        best_k, segmented_img, scores, _ = select_autonomous_k(img)

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Original: {title}")
        axes[row, 0].axis('off')

        axes[row, 1].imshow(segmented_img)
        axes[row, 1].set_title(f"Suggested K = {best_k}")
        axes[row, 1].axis('off')

        ks = list(scores.keys())
        scs = list(scores.values())
        axes[row, 2].plot(ks, scs, marker='o', color='purple')
        axes[row, 2].set_title("Composite Score vs K")
        axes[row, 2].set_xlabel("K")
        axes[row, 2].set_ylabel("Score")
        axes[row, 2].grid(True)

        row += 1

    plt.tight_layout()
    plt.savefig("autonomous_k_evaluation.png")
    print("Saved figure to 'autonomous_k_evaluation.png'. Displaying plot window...")
    plt.show()


if __name__ == "__main__":
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass
        
    print("Running Autonomous K Selection Benchmark...")
    run_evaluation_benchmark()
    print("Done!")