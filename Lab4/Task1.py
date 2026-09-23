import os
import glob
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern, hog
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import normalize
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ==============================================================================
# 1. FEATURE EXTRACTION FUNCTIONS
# ==============================================================================

def extract_color_histogram(image_bgr, bins=(8, 8, 8)):
    """Extracts a normalized 3D color histogram in HSV space."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_lbp_histogram(image_gray, num_points=24, radius=3, bins=26):
    """Extracts a normalized Local Binary Pattern (LBP) texture histogram."""
    lbp = local_binary_pattern(image_gray, num_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
    return hist


def extract_hog_features(image_gray, target_size=(128, 128)):
    """Extracts HOG features for global shape and orientation structure."""
    resized = cv2.resize(image_gray, target_size)
    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=False
    )
    return features


def extract_all_features_separately(images_bgr, kmeans_bvw):
    """
    Extracts feature vectors for each modality:
    - SIFT Bag-of-Visual-Words
    - HSV Color Histogram
    - LBP Texture Histogram
    - HOG Shape Descriptor
    
    Applies independent L2-normalization to each feature block before concatenation.
    """
    bvw_feats, color_feats, lbp_feats, hog_feats = [], [], [], []
    sift = cv2.SIFT_create()

    for img_bgr in images_bgr:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. BoVW (SIFT)
        kp, des = sift.detectAndCompute(img_gray, None)
        bvw_hist = np.zeros(kmeans_bvw.n_clusters)
        if des is not None:
            words = kmeans_bvw.predict(des)
            np.add.at(bvw_hist, words, 1)
        bvw_feats.append(bvw_hist)

        # 2. HSV Color Histogram
        color_feats.append(extract_color_histogram(img_bgr))

        # 3. LBP Texture Histogram
        lbp_feats.append(extract_lbp_histogram(img_gray))

        # 4. HOG Shape Feature
        hog_feats.append(extract_hog_features(img_gray))

    # Independent L2 Normalization per feature family
    X_bvw = normalize(np.array(bvw_feats), norm='l2')
    X_color = normalize(np.array(color_feats), norm='l2')
    X_lbp = normalize(np.array(lbp_feats), norm='l2')
    X_hog = normalize(np.array(hog_feats), norm='l2')

    X_combined = np.hstack([X_bvw, X_color, X_lbp, X_hog])
    
    return X_bvw, X_combined


# ==============================================================================
# 2. HIERARCHICAL CLASSIFIER IMPLEMENTATION
# ==============================================================================

class HierarchicalClassifierPipeline:
    """
    Two-Stage Hierarchical Classifier:
    1. Coarse SGD-SVM predicts visual supergroup ID.
    2. Fine SGD-SVMs (one per supergroup) predict the exact target class.
    """
    def __init__(self, n_supergroups=16, random_state=42):
        self.n_supergroups = n_supergroups
        self.random_state = random_state
        self.coarse_clf = SGDClassifier(
            loss='hinge', penalty='l2', warm_start=True, random_state=random_state
        )
        self.fine_clfs = {}
        self.class_to_group = {}
        self.group_to_classes = {}

    def build_hierarchy(self, X_train, y_train):
        unique_classes = np.unique(y_train)
        actual_supergroups = min(self.n_supergroups, len(unique_classes))

        if actual_supergroups < 2:
            self.class_to_group = {c: 0 for c in unique_classes}
            self.group_to_classes = {0: list(unique_classes)}
            return

        centroids = np.array([X_train[y_train == c].mean(axis=0) for c in unique_classes])

        clusterer = AgglomerativeClustering(
            n_clusters=actual_supergroups, metric='euclidean', linkage='ward'
        )
        group_assignments = clusterer.fit_predict(centroids)

        self.class_to_group = {}
        self.group_to_classes = {}
        for cls, grp in zip(unique_classes, group_assignments):
            self.class_to_group[cls] = grp
            self.group_to_classes.setdefault(grp, []).append(cls)

    def fit_with_loss(self, X_train, y_train, X_val, y_val, epochs=40):
        self.build_hierarchy(X_train, y_train)
        y_train_coarse = np.array([self.class_to_group[c] for c in y_train])
        y_val_coarse = np.array([self.class_to_group[c] for c in y_val])

        # 1. Train Coarse Classifier
        coarse_tr_loss, coarse_va_loss = [], []
        unique_groups = np.unique(y_train_coarse)

        if len(unique_groups) > 1:
            for epoch in range(epochs):
                self.coarse_clf.partial_fit(X_train, y_train_coarse, classes=unique_groups)
                coarse_tr_loss.append(1.0 - self.coarse_clf.score(X_train, y_train_coarse))
                coarse_va_loss.append(1.0 - self.coarse_clf.score(X_val, y_val_coarse))

        # 2. Train Fine Classifiers per Supergroup
        fine_histories = {}
        for grp, classes in self.group_to_classes.items():
            if len(classes) > 1:
                mask_tr = np.isin(y_train, classes)
                mask_va = np.isin(y_val, classes)

                clf = SGDClassifier(
                    loss='hinge', penalty='l2', warm_start=True, random_state=self.random_state
                )
                tr_loss, va_loss = [], []

                for epoch in range(epochs):
                    clf.partial_fit(X_train[mask_tr], y_train[mask_tr], classes=np.array(classes))
                    tr_loss.append(1.0 - clf.score(X_train[mask_tr], y_train[mask_tr]))
                    if mask_va.sum() > 0:
                        va_loss.append(1.0 - clf.score(X_val[mask_va], y_val[mask_va]))
                    else:
                        va_loss.append(0.0)

                self.fine_clfs[grp] = clf
                fine_histories[grp] = (tr_loss, va_loss)
            else:
                self.fine_clfs[grp] = classes[0]

        return (coarse_tr_loss, coarse_va_loss), fine_histories

    def predict(self, X):
        if not hasattr(self.coarse_clf, "classes_"):
            single_grp = list(self.group_to_classes.keys())[0]
            coarse_preds = np.full(len(X), single_grp)
        else:
            coarse_preds = self.coarse_clf.predict(X)

        final_preds = np.zeros(len(X), dtype=int)
        for grp in np.unique(coarse_preds):
            idx = np.where(coarse_preds == grp)[0]
            if grp in self.fine_clfs:
                model_or_label = self.fine_clfs[grp]
                if isinstance(model_or_label, SGDClassifier):
                    final_preds[idx] = model_or_label.predict(X[idx])
                else:
                    final_preds[idx] = model_or_label
        return final_preds


def train_sgd_svm_with_loss(X_train, y_train, X_val, y_val, all_classes, epochs=40, random_state=42):
    clf = SGDClassifier(loss='hinge', penalty='l2', warm_start=True, random_state=random_state)
    train_loss, val_loss = [], []

    for epoch in range(epochs):
        clf.partial_fit(X_train, y_train, classes=all_classes)
        train_loss.append(1.0 - clf.score(X_train, y_train))
        val_loss.append(1.0 - clf.score(X_val, y_val))

    return clf, train_loss, val_loss


# ==============================================================================
# 3. MAIN EXECUTION WITH DATASET LOADER
# ==============================================================================

if __name__ == "__main__":
    print("==================================================")
    print(" TASK 1: FEATURE ENGINEERING & HIERARCHICAL MODEL")
    print("==================================================")

    # Path to dataset folder
    dataset_dir = r"E:\CV-LabAssignments\Lab4\256_ObjectCategories"  # Update path if your folder has a different name
    
    images, labels, class_names = [], [], []
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    if os.path.exists(dataset_dir):
        category_dirs = sorted([
            d for d in os.listdir(dataset_dir) 
            if os.path.isdir(os.path.join(dataset_dir, d))
        ])
        print(f"Found {len(category_dirs)} class folders in '{dataset_dir}'. Loading images...")

        # Number of images to load per class (increase for full run)
        max_images_per_class = 15

        for class_idx, cat_name in enumerate(category_dirs):
            class_names.append(cat_name)
            cat_path = os.path.join(dataset_dir, cat_name)
            img_files = [f for f in os.listdir(cat_path) if f.lower().endswith(valid_extensions)]
            
            for img_file in img_files[:max_images_per_class]:
                img_path = os.path.join(cat_path, img_file)
                img_bgr = cv2.imread(img_path)
                if img_bgr is not None:
                    img_resized = cv2.resize(img_bgr, (128, 128))
                    images.append(img_resized)
                    labels.append(class_idx)

        images = np.array(images)
        labels = np.array(labels)
        n_classes = len(class_names)
        print(f"Loaded {len(images)} total images across {n_classes} classes.")
    else:
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        exit(1)

    # Train/Val/Test Split (70/15/15)
    np.random.seed(42)
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    
    train_cutoff = int(0.70 * len(labels))
    val_cutoff = int(0.85 * len(labels))

    idx_tr = indices[:train_cutoff]
    idx_va = indices[train_cutoff:val_cutoff]
    idx_te = indices[val_cutoff:]

    imgs_tr, y_tr = images[idx_tr], labels[idx_tr]
    imgs_va, y_va = images[idx_va], labels[idx_va]
    imgs_te, y_te = images[idx_te], labels[idx_te]

    # Fit SIFT Visual Dictionary
    print("Building SIFT Bag-of-Visual-Words vocabulary...")
    sift = cv2.SIFT_create()
    all_descriptors = []
    for img in imgs_tr:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, des = sift.detectAndCompute(gray, None)
        if des is not None:
            all_descriptors.append(des)

    all_descriptors = np.vstack(all_descriptors) if len(all_descriptors) > 0 else np.random.randn(100, 128).astype(np.float32)
    kmeans_bvw = KMeans(n_clusters=64, random_state=42, n_init=5).fit(all_descriptors)

    # Feature Extraction
    print("Extracting features for Train, Validation, and Test sets...")
    X_tr_bvw, X_tr_comb = extract_all_features_separately(imgs_tr, kmeans_bvw)
    X_va_bvw, X_va_comb = extract_all_features_separately(imgs_va, kmeans_bvw)
    X_te_bvw, X_te_comb = extract_all_features_separately(imgs_te, kmeans_bvw)

    all_classes = np.unique(labels)
    epochs = 40

    # Model Training
    print("\n--- Training Model (a): SIFT-BoVW Flat Baseline ---")
    clf_a, tr_loss_a, va_loss_a = train_sgd_svm_with_loss(
        X_tr_bvw, y_tr, X_va_bvw, y_va, all_classes, epochs=epochs
    )
    preds_a = clf_a.predict(X_te_bvw)

    print("\n--- Training Model (b): Combined-Features Flat Classifier ---")
    clf_b, tr_loss_b, va_loss_b = train_sgd_svm_with_loss(
        X_tr_comb, y_tr, X_va_comb, y_va, all_classes, epochs=epochs
    )
    preds_b = clf_b.predict(X_te_comb)

    print("\n--- Training Model (c): Two-Stage Hierarchical Classifier ---")
    hier_pipeline = HierarchicalClassifierPipeline(n_supergroups=16, random_state=42)
    (coarse_tr_loss, coarse_va_loss), fine_histories = hier_pipeline.fit_with_loss(
        X_tr_comb, y_tr, X_va_comb, y_va, epochs=epochs
    )
    preds_c = hier_pipeline.predict(X_te_comb)

    # Save artifacts
    artifacts = {
        "kmeans_bvw": kmeans_bvw,
        "classifier": clf_b,
        "class_names": class_names
    }
    with open("best_pipeline.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print("\nSaved best model artifacts to 'best_pipeline.pkl'.")

    # Evaluation
    def calc_metrics(y_true, y_pred):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        acc = accuracy_score(y_true, y_pred)
        return acc, p, r, f1

    acc_a, p_a, r_a, f1_a = calc_metrics(y_te, preds_a)
    acc_b, p_b, r_b, f1_b = calc_metrics(y_te, preds_b)
    acc_c, p_c, r_c, f1_c = calc_metrics(y_te, preds_c)

    print("\n" + "="*75)
    print("                         FINAL TEST RESULTS")
    print("="*75)
    print(f"{'Model Pipeline':<38} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 75)
    print(f"{'(a) SIFT-only Flat Baseline':<38} | {acc_a*100:6.2f}%  | {p_a:9.4f} | {r_a:8.4f} | {f1_a:8.4f}")
    print(f"{'(b) Improved Combined Features Flat':<38} | {acc_b*100:6.2f}%  | {p_b:9.4f} | {r_b:8.4f} | {f1_b:8.4f}")
    print(f"{'(c) Hierarchical Classifier':<38} | {acc_c*100:6.2f}%  | {p_c:9.4f} | {r_c:8.4f} | {f1_c:8.4f}")
    print("="*75)

    # Plot Loss Curves
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(tr_loss_a, label='Train Error')
    plt.plot(va_loss_a, label='Val Error')
    plt.title('(a) Baseline SIFT Flat Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Misclassification Rate')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(tr_loss_b, label='Train Error')
    plt.plot(va_loss_b, label='Val Error')
    plt.title('(b) Combined Features Flat Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(coarse_tr_loss, label='Coarse Train Error')
    plt.plot(coarse_va_loss, label='Coarse Val Error')
    plt.title('(c) Hierarchical Coarse Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("loss_curves.png")
    print("\nSaved loss curves plot to 'loss_curves.png'.")
    plt.show()