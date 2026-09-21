import cv2
import numpy as np

# ---- Hardcoded configuration: change these to experiment ----
FEATURE = "AKAZE"     # options: "SIFT", "ORB", "AKAZE"
MATCHER = "FLANN"    # options: "BF", "FLANN"
RATIO_TEST_THRESHOLD = 0.75

img1 = cv2.imread("hpatches-sequences-release/v_wall/1.ppm")
img2 = cv2.imread("hpatches-sequences-release/v_wall/2.ppm")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# --- Build the chosen feature detector/descriptor ---
if FEATURE == "SIFT":
    detector = cv2.SIFT_create()
    is_binary_descriptor = False
elif FEATURE == "ORB":
    detector = cv2.ORB_create(nfeatures=1000)
    is_binary_descriptor = True
elif FEATURE == "AKAZE":
    detector = cv2.xfeatures2d.AKAZE_create()
    is_binary_descriptor = True
else:
    raise ValueError(f"Unknown FEATURE: {FEATURE}")

kp1, des1 = detector.detectAndCompute(gray1, None)
kp2, des2 = detector.detectAndCompute(gray2, None)

# --- Build the chosen matcher ---
if MATCHER == "BF":
    norm_type = cv2.NORM_HAMMING if is_binary_descriptor else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm_type)
elif MATCHER == "FLANN":
    if is_binary_descriptor:
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)  # FLANN_INDEX_LSH
    else:
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
else:
    raise ValueError(f"Unknown MATCHER: {MATCHER}")

# --- Match with Lowe's ratio test: keep a match only if the best candidate
#     is meaningfully closer than the second-best (filters ambiguous matches) ---
knn_matches = matcher.knnMatch(des1, des2, k=2)
good_matches = []
for pair in knn_matches:
    if len(pair) == 2:
        m, n = pair
        if m.distance < RATIO_TEST_THRESHOLD * n.distance:
            good_matches.append(m)

print(f"{FEATURE} + {MATCHER}: {len(good_matches)} good matches out of {len(knn_matches)} candidates")

match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.namedWindow("Feature Matches", cv2.WINDOW_NORMAL)
cv2.imshow("Feature Matches", match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()