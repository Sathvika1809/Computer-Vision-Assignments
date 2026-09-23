import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib

from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# ---------------- 0) Configuration ----------------
CALTECH_DIR = r"E:\CV-LabAssignments\Lab4\256_ObjectCategories" # Update to your exact path
VOCAB_SIZE = 64                        # k for the visual-vocabulary K-Means
RESIZE_DIM = 128                       # resize every image to a consistent size
MAX_IMAGES_PER_CLASS = 15              # Subsample per class for lab speed
NUM_EPOCHS = 30                        # SGD-SVM training epochs
RANDOM_STATE = 42

# ---------------- 1) DataLoader with Auto-Fallback ----------------
def load_caltech256_paths(root_dir, max_per_class):
    """Walks Caltech-256 class directories and returns file paths and labels."""
    if not os.path.exists(root_dir):
        # Fallback search if path is nested
        parent_dir = os.path.dirname(root_dir)
        possible_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if "256" in d]
        if possible_dirs:
            root_dir = possible_dirs[0]
            print(f"Auto-located dataset directory: '{root_dir}'")
        else:
            raise FileNotFoundError(f"Dataset directory '{root_dir}' not found.")

    class_folders = sorted(
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )
    label_names = [name.split(".", 1)[1] if "." in name else name for name in class_folders]

    rng = random.Random(RANDOM_STATE)
    filepaths, labels = [], []

    for label_id, folder in enumerate(class_folders):
        folder_path = os.path.join(root_dir, folder)
        images_in_class = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        rng.shuffle(images_in_class)
        for fname in images_in_class[:max_per_class]:
            filepaths.append(os.path.join(folder_path, fname))
            labels.append(label_id)

    return filepaths, np.array(labels), label_names

print("Loading dataset file paths...")
filepaths, labels, label_names = load_caltech256_paths(CALTECH_DIR, MAX_IMAGES_PER_CLASS)
print(f"Loaded {len(filepaths)} images across {len(label_names)} classes.")

# ---------------- 2) Train / Val / Test Split ----------------
# Non-stratified fallback if class samples are too small for stratified splitting
try:
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        filepaths, labels, test_size=0.30, stratify=labels, random_state=RANDOM_STATE
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=RANDOM_STATE
    )
except ValueError:
    print("Warning: Sample size per class too small for stratified split. Using standard random split.")
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        filepaths, labels, test_size=0.30, random_state=RANDOM_STATE
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=RANDOM_STATE
    )

print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

# ---------------- 3) SIFT Descriptors Extraction ----------------
sift = cv2.SIFT_create()

def extract_sift_descriptors(paths):
    all_descriptors = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            all_descriptors.append(None)
            continue
        resized = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, descriptors = sift.detectAndCompute(gray, None)
        all_descriptors.append(descriptors)
    return all_descriptors

print("Extracting SIFT descriptors for Train set...")
train_descriptors = extract_sift_descriptors(train_paths)
print("Extracting SIFT descriptors for Validation set...")
val_descriptors = extract_sift_descriptors(val_paths)
print("Extracting SIFT descriptors for Test set...")
test_descriptors = extract_sift_descriptors(test_paths)

# ---------------- 4) K-Means Vocabulary ----------------
valid_train_des = [d for d in train_descriptors if d is not None]
if len(valid_train_des) == 0:
    raise ValueError("No SIFT features found in training set. Check image paths.")

all_train_descriptors = np.vstack(valid_train_des)
print(f"Total training SIFT descriptors: {all_train_descriptors.shape[0]}")
print(f"Clustering into a {VOCAB_SIZE}-word visual vocabulary...")

vocabulary = KMeans(n_clusters=VOCAB_SIZE, n_init=4, random_state=RANDOM_STATE)
vocabulary.fit(all_train_descriptors)

# ---------------- 5) Bag-of-Visual-Words Features ----------------
def to_bovw_histogram(descriptors, vocabulary, vocab_size):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(vocab_size, dtype=np.float32)

    word_ids = vocabulary.predict(descriptors)
    histogram, _ = np.histogram(word_ids, bins=np.arange(vocab_size + 1))
    histogram = histogram.astype(np.float32)

    norm = np.linalg.norm(histogram)
    if norm > 0:
        histogram /= norm
    return histogram

def build_feature_matrix(descriptor_list, vocabulary, vocab_size):
    return np.array([to_bovw_histogram(d, vocabulary, vocab_size) for d in descriptor_list])

X_train = build_feature_matrix(train_descriptors, vocabulary, VOCAB_SIZE)
X_val = build_feature_matrix(val_descriptors, vocabulary, VOCAB_SIZE)
X_test = build_feature_matrix(test_descriptors, vocabulary, VOCAB_SIZE)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ---------------- 6) Fast Vectorized Multiclass Hinge Loss ----------------
def fast_multiclass_hinge_loss(decision_values, true_labels, all_classes):
    """Vectorized calculation of multi-class hinge loss."""
    n_samples, n_classes = decision_values.shape
    # Create one-hot encoding (-1 for wrong class, +1 for correct class)
    y_bin = np.full((n_samples, n_classes), -1)
    
    # Map class labels to index columns
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    label_indices = np.array([class_to_idx[lbl] for lbl in true_labels])
    y_bin[np.arange(n_samples), label_indices] = 1

    margins = np.maximum(0, 1 - y_bin * decision_values)
    return float(np.mean(margins))

svm = SGDClassifier(loss="hinge", random_state=RANDOM_STATE, learning_rate="optimal")
all_classes = np.unique(labels)

train_losses, val_losses = [], []
print("\nTraining SGD-SVM Classifier...")

for epoch in range(NUM_EPOCHS):
    svm.partial_fit(X_train, train_labels, classes=all_classes)

    tr_dec = svm.decision_function(X_train)
    va_dec = svm.decision_function(X_val)

    train_loss = fast_multiclass_hinge_loss(tr_dec, train_labels, all_classes)
    val_loss = fast_multiclass_hinge_loss(va_dec, val_labels, all_classes)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# Plotting Loss Curves
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Hinge Loss")
plt.title("Baseline Section 2: SVM (SGD) Training/Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("baseline_training_curve.png")
print("\nSaved loss curve to 'baseline_training_curve.png'.")
plt.show()

# ---------------- 7) Evaluation & Output ----------------
test_predictions = svm.predict(X_test)
test_acc = accuracy_score(test_labels, test_predictions)

print(f"\n==========================================")
print(f"Baseline Test Set Accuracy: {test_acc * 100:.2f}%")
print(f"==========================================\n")

report = classification_report(
    test_labels, test_predictions, target_names=label_names, zero_division=0
)
print("Classification Report:\n")
print(report)

# ---------------- 8) Save Model Pipeline ----------------
joblib.dump(svm, "caltech256_svm.joblib")
joblib.dump(vocabulary, "caltech256_vocabulary.joblib")
joblib.dump(scaler, "caltech256_scaler.joblib")
joblib.dump(label_names, "caltech256_label_names.joblib")
print("Successfully saved model, vocabulary, scaler, and label names to disk.")