"""
Lab 4 - Classify a single hardcoded image using the SVM + vocabulary saved
by the Caltech-256 training script above. Unlike that script, this one
DOES display the image, since it's meant for one-off, visual inspection.
"""

import cv2
import joblib
import numpy as np

IMAGE_PATH = "my_test_image.jpg"  # hardcoded path - change to whatever you want to classify
RESIZE_DIM = 128
VOCAB_SIZE = 64

svm = joblib.load("caltech256_svm.joblib")
vocabulary = joblib.load("caltech256_vocabulary.joblib")
scaler = joblib.load("caltech256_scaler.joblib")
label_names = joblib.load("caltech256_label_names.joblib")

sift = cv2.SIFT_create()

img = cv2.imread(IMAGE_PATH)
resized = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
_, descriptors = sift.detectAndCompute(gray, None)

if descriptors is None:
    histogram = np.zeros(VOCAB_SIZE, dtype=np.float32)
else:
    word_ids = vocabulary.predict(descriptors)
    histogram, _ = np.histogram(word_ids, bins=np.arange(VOCAB_SIZE + 1))
    histogram = histogram.astype(np.float32)
    norm = np.linalg.norm(histogram)
    if norm > 0:
        histogram /= norm

feature_vector = scaler.transform(histogram.reshape(1, -1))
predicted_class_id = svm.predict(feature_vector)[0]
predicted_label = label_names[predicted_class_id]

print(f"Predicted class: {predicted_label}")

display_img = img.copy()
cv2.putText(display_img, predicted_label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
cv2.imshow("Prediction", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()