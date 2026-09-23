import cv2
import pickle
import numpy as np

def run_single_image_inference(image_path, pipeline_dict_path="best_pipeline.pkl"):
    """
    Inference script for the best-performing pipeline (Combined Features Flat Classifier).
    """
    # Load model state
    with open(pipeline_dict_path, "rb") as f:
        artifacts = pickle.load(f)
    
    kmeans_bvw = artifacts["kmeans_bvw"]
    scaler = artifacts["scaler"]
    classifier = artifacts["classifier"]
    class_names = artifacts["class_names"]

    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    # Feature Extraction
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. SIFT-BoVW
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_gray, None)
    bvw_hist = np.zeros((1, kmeans_bvw.n_clusters))
    if des is not None:
        words = kmeans_bvw.predict(des)
        np.add.at(bvw_hist[0], words, 1)

    # 2. Color Histogram
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    color_hist = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
    cv2.normalize(color_hist, color_hist)
    color_feats = color_hist.flatten().reshape(1, -1)

    # 3. LBP Texture
    from skimage.feature import local_binary_pattern, hog
    lbp = local_binary_pattern(img_gray, 24, 3, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
    lbp_feats = lbp_hist.reshape(1, -1)

    # 4. HOG Shape
    resized = cv2.resize(img_gray, (128, 128))
    hog_feats = hog(resized, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), visualize=False).reshape(1, -1)

    # Normalize independently & Concatenate
    X_bvw = normalize(bvw_hist, norm='l2')
    X_color = normalize(color_feats, norm='l2')
    X_lbp = normalize(lbp_feats, norm='l2')
    X_hog = normalize(hog_feats, norm='l2')
    
    X_combined = np.hstack([X_bvw, X_color, X_lbp, X_hog])
    X_scaled = scaler.transform(X_combined)

    # Inference
    pred_idx = classifier.predict(X_scaled)[0]
    predicted_label = class_names[pred_idx]

    print(f"======================================")
    print(f"Input Image : {image_path}")
    print(f"Prediction  : {predicted_label}")
    print(f"======================================")
    return predicted_label

# Example invocation:
# run_single_image_inference("test_samples/001_0001.jpg")