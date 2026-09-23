import cv2
import numpy as np
from sklearn.cluster import KMeans

IMAGE_PATH = "sipi-dataset/misc/misc/4.1.05.tiff"  # hardcoded path - change as needed

img = cv2.imread(IMAGE_PATH)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert img to HSV

h, w = img.shape[:2] 
hsv_pixels = hsv.reshape(-1, 3).astype(np.float32)

# A roughly 3-pixel-diameter circular structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Make it a kernel and apply on an image into functions like erosion,dilation,opening and closing

def segment_image(k):
    kmeans = KMeans(n_clusters=k, n_init=4, random_state=42)
    labels = kmeans.fit_predict(hsv_pixels)
    label_map = labels.reshape(h, w)

    output = np.zeros_like(img)
    for cluster_id in range(k):
        mask = (label_map == cluster_id).astype(np.uint8) * 255
        # Close small holes/gaps within this color's mask before averaging
        # We made kernel at starting and then use it acc to what we need
        
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        bool_mask = closed_mask > 0
        if np.any(bool_mask):
            mean_color = img[bool_mask].mean(axis=0)  # mean RGB (BGR order), from the ORIGINAL image
            output[bool_mask] = mean_color

    return output


def on_trackbar(k):
    k = max(k, 2)  # guard in case a backend allows the slider to hit 0/1 before setTrackbarMin kicks in
    output = segment_image(k)
    cv2.imshow("Final Segmented Output", output)


cv2.namedWindow("HSV Output", cv2.WINDOW_NORMAL)
cv2.imshow("HSV Output", hsv)  # NOTE: displayed as if it were BGR, so this looks like "false colors" -
                                # that's expected, it's just a raw look at the data K-Means is clustering on

cv2.namedWindow("Final Segmented Output", cv2.WINDOW_NORMAL)
cv2.createTrackbar("K", "Final Segmented Output", 2, 20, on_trackbar)
cv2.setTrackbarMin("K", "Final Segmented Output", 2)  # keeps the slider from going below 2

on_trackbar(2)  # initial render at the default K=2

cv2.waitKey(0)
cv2.destroyAllWindows()