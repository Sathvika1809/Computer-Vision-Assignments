#Exploring Edge Detection
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sipi-dataset/misc/misc/4.1.05.tiff") #pick any image with clear edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(3,3),0) #mild denoising before edge detection
# here (3,3) represents kernel size and 0 represents Gaussian standard Deviation(Sigma X)

# Sobel
sobel_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
sobel_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
sobel_magnitude = cv2.magnitude(sobel_x,sobel_y)
sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude) # Converts 64 bit to 8 bit

#Laplacian of Gaussian (LoG)
blurred_for_log = cv2.GaussianBlur(gray,(5,5), 0) # kernel(5,5), 0 st.dev
log = cv2.Laplacian(blurred_for_log, cv2.CV_64F,ksize=3)
log = cv2.convertScaleAbs(log)

#Canny
canny = cv2.Canny(gray,100,200)
# Here 100 and 200 are the low and high thresholds used 
# for hysteresis thresholding to decide which edges are kept 
# or discarded
fig, axes = plt.subplots(2,3, figsize=(15,8))
titles = ["Original (Gray)","Sobel X","Sobel Y","Sobel Magnitude","Laplacian of Gaussian","Canny"]
images = [gray, cv2.convertScaleAbs(sobel_x),cv2.convertScaleAbs(sobel_y),sobel_magnitude,log,canny]

for ax, title, im in zip(axes.ravel(),titles,images):
    ax.imshow(im,cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()

img = cv2.imread("sipi-dataset/misc/misc/4.1.05.tiff")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)

def nothing(x):
    pass

cv2.namedWindow("Canny Threshold Explorer", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Low Threshold", "Canny Threshold Explorer", 50, 500, nothing)
cv2.createTrackbar("High Threshold", "Canny Threshold Explorer", 150, 500, nothing)

while True:
    low = cv2.getTrackbarPos("Low Threshold", "Canny Threshold Explorer")
    high = cv2.getTrackbarPos("High Threshold", "Canny Threshold Explorer")

    edges = cv2.Canny(gray, low, high)
    cv2.imshow("Canny Threshold Explorer", edges)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
        break

cv2.destroyAllWindows()

# Morphological Operations
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# Erosion - shrinks white(foreground) regions - A pixel
# stays white only if every pixel under the kernel is white, otherwise
# it's set to black
# Purpose: remove small white noise specks, detach weakl-connected
# objects , thin out object boundaries
# Trade-off: it also shrinks and can break apart legitimate thin structure
# if overused

# Comparing Erosion,Dilation,Opening,Closing

img = cv2.imread("sipi-dataset/misc/misc/4.1.05.tiff")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# otsu algo cutoff point is determined instead of manual hardcoded number
# _ is the placeholder for the first return value which stores optimal threshold value which otsu calculated
# binary is the final opt img where every pixel is strictly 0(black) or 255(white)



kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

eroded = cv2.erode(binary, kernel, iterations=1)
dilated = cv2.dilate(binary, kernel, iterations=1)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
titles = ["Binary (Otsu)", "Erosion", "Dilation", "Opening", "Closing"]
images = [binary, eroded, dilated, opened, closed]

for ax, title, im in zip(axes, titles, images):
    ax.imshow(im, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
# Contour Detection and Polygon Approximation
import cv2
import numpy as np

img = cv2.imread("your_object_image.jpg")   # any image with one clear foreground object
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Segment foreground from background
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Clean up the mask: opening removes small noise, closing seals small gaps
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours on the cleaned mask
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

result = img.copy()
cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)

# Polygon approximation: simplify the contour to fewer vertices
epsilon = 0.01 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)
cv2.drawContours(result, [approx], -1, (0, 0, 255), 2)

cv2.imshow("Contour (green) vs Approximated Polygon (red)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


img = cv2.imread("your_circles_image.jpg")  # e.g. coins, balls, wheels — any image with round objects
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)  # reduces false circle detections from noise

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,             # inverse ratio of accumulator resolution to image resolution
    minDist=30,        # minimum distance between detected circle centers
    param1=100,        # higher Canny threshold used internally
    param2=40,         # accumulator threshold — lower means more (possibly false) circles
    minRadius=10,
    maxRadius=100,
)

result = img.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0, :]:
        cv2.circle(result, (x, y), r, (0, 255, 0), 2)   # circle outline
        cv2.circle(result, (x, y), 2, (0, 0, 255), 3)   # circle center

cv2.imshow("Detected Circles", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread("your_lines_image.jpg")  # e.g. a building, a page/document, a road — anything with straight edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# ---- Standard Hough Transform: full (infinite) lines ----
lines_img = img.copy()
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)

if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Extend far beyond the image bounds in both directions along the line
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# ---- Probabilistic Hough Transform: line segments ----
segments_img = img.copy()
segments = cv2.HoughLinesP(
    edges, 1, np.pi / 180, threshold=80,
    minLineLength=50, maxLineGap=10
)

if segments is not None:
    for x1, y1, x2, y2 in segments[:, 0]:
        cv2.line(segments_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Edges", edges)
cv2.imshow("Standard Hough Lines (HoughLines)", lines_img)
cv2.imshow("Probabilistic Hough Lines (HoughLinesP)", segments_img)
cv2.waitKey(0)
cv2.destroyAllWindows()