import cv2
import numpy as np

img = cv2.imread("hpatches-sequences-release/v_wall/1.ppm")  # hardcoded path — change as needed
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

RICH = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # draws circle sized to keypoint.size + orientation tick

# ---- Harris corners (no descriptor, no scale/orientation) ----
gray_f = np.float32(gray)
harris_response = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
harris_response = cv2.dilate(harris_response, None)  # helps mark local maxima more visibly
harris_img = img.copy()
threshold = 0.01 * harris_response.max()
ys, xs = np.where(harris_response > threshold)
for x, y in zip(xs, ys):
    cv2.circle(harris_img, (int(x), int(y)), 3, (0, 0, 255), 1)

# ---- SIFT ----
sift = cv2.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(gray, None)
sift_img = cv2.drawKeypoints(img, kp_sift, None, color=(0, 255, 0), flags=RICH)

# ---- FAST (no scale/orientation of its own) ----
fast = cv2.FastFeatureDetector_create(threshold=25)
kp_fast = fast.detect(gray, None)
fast_img = cv2.drawKeypoints(img, kp_fast, None, color=(255, 0, 0), flags=RICH)

# ---- BRIEF (descriptor only — paired here with a STAR detector for keypoints) ----
star = cv2.xfeatures2d.StarDetector_create()
kp_star = star.detect(gray, None)
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
kp_brief, des_brief = brief.compute(gray, kp_star)
brief_img = cv2.drawKeypoints(img, kp_brief, None, color=(0, 255, 255), flags=RICH)

# ---- ORB ----
orb = cv2.ORB_create(nfeatures=500)
kp_orb, des_orb = orb.detectAndCompute(gray, None)
orb_img = cv2.drawKeypoints(img, kp_orb, None, color=(255, 0, 255), flags=RICH)

# ---- AKAZE ----
akaze = cv2.xfeatures2d.AKAZE_create()
kp_akaze, des_akaze = akaze.detectAndCompute(gray, None)
akaze_img = cv2.drawKeypoints(img, kp_akaze, None, color=(0, 128, 255), flags=RICH)

# ---- FREAK (descriptor only — paired here with AKAZE for scale+orientation) ----
kp_for_freak = akaze.detect(gray, None)
freak = cv2.xfeatures2d.FREAK_create()
kp_freak, des_freak = freak.compute(gray, kp_for_freak)
freak_img = cv2.drawKeypoints(img, kp_freak, None, color=(128, 0, 255), flags=RICH)

windows = {
    "Harris Corners": harris_img,
    "SIFT": sift_img,
    "FAST": fast_img,
    "BRIEF (on STAR keypoints)": brief_img,
    "ORB": orb_img,
    "AKAZE": akaze_img,
    "FREAK (on AKAZE keypoints)": freak_img,
}

for name, im in windows.items():
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, im)

cv2.waitKey(0)
cv2.destroyAllWindows()