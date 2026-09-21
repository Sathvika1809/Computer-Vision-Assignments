import os
import glob
import cv2
import numpy as np

# =====================================================================
# CONFIGURATION & HYPERPARAMETERS
# =====================================================================
FEATURE_TYPE = 'SIFT'        # Switching to SIFT handles difficult scale/perspective better
MAX_FEATURES = 3000          # Maximum features per image (for ORB)
RATIO_TEST_THRESH = 0.70     # Lowe's ratio test threshold
RANSAC_REPROJ_THRESH = 3.0   # Reprojection threshold in pixels
MIN_INLIER_COUNT = 25        # Minimum inliers required to form a graph edge

MAX_CANVAS_DIM = 12000       # Maximum canvas width/height limit (pixels)
OUTPUT_CANVAS_COLOR = (0, 0, 0) # Background color
# =====================================================================


def detect_and_compute(image, feature_type='SIFT'):
    """Extract keypoints and descriptors from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if feature_type.upper() == 'SIFT':
        detector = cv2.SIFT_create()
    elif feature_type.upper() == 'ORB':
        detector = cv2.ORB_create(nfeatures=MAX_FEATURES)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2, feature_type='SIFT', ratio_thresh=0.70):
    """Perform KNN matching and apply Lowe's ratio test."""
    if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
        return []

    norm_type = cv2.NORM_HAMMING if feature_type.upper() == 'ORB' else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for match_tuple in knn_matches:
        if len(match_tuple) == 2:
            m, n = match_tuple
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
                
    return good_matches


def is_valid_homography(H, max_scale=4.0, min_det=0.05, max_det=20.0):
    """
    Sanity check to reject ill-conditioned homographies that cause
    extreme canvas blowing and perspective drift.
    """
    if H is None:
        return False

    # 1. Determinant check (avoids zero/collapsed transformations)
    det = np.linalg.det(H)
    if det < min_det or det > max_det:
        return False

    # 2. Scale factor check using Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(H[:2, :2])
    if S[1] < 1e-6:
        return False
        
    scale_ratio = S[0] / S[1]
    if scale_ratio > max_scale:
        return False

    return True


def estimate_homography_ransac(kp1, kp2, matches, reproj_thresh=3.0):
    """Estimate Homography using RANSAC and return inliers."""
    if len(matches) < 4:
        return None, 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_thresh)
    if mask is None:
        return None, 0

    inlier_count = int(np.sum(mask))
    return H, inlier_count


def build_image_graph(images, keypoints_list, descriptors_list):
    """Compute pairwise matches and construct an adjacency graph."""
    n = len(images)
    adj_matrix = {i: {} for i in range(n)}
    
    for i in range(n):
        for j in range(i + 1, n):
            matches = match_descriptors(
                descriptors_list[i], descriptors_list[j], 
                feature_type=FEATURE_TYPE, ratio_thresh=RATIO_TEST_THRESH
            )
            H, inliers = estimate_homography_ransac(
                keypoints_list[i], keypoints_list[j], matches, 
                reproj_thresh=RANSAC_REPROJ_THRESH
            )

            # Apply both inlier minimum and homography sanity check
            if inliers >= MIN_INLIER_COUNT and is_valid_homography(H):
                adj_matrix[i][j] = (H, inliers)
                adj_matrix[j][i] = (np.linalg.inv(H), inliers)

    return adj_matrix


def filter_and_chain_transforms(adj_matrix, num_images):
    """Chain homographies using Maximum Inlier tree traversal relative to anchor image."""
    accepted_indices = [i for i in adj_matrix if len(adj_matrix[i]) > 0]
    if not accepted_indices:
        return None, list(range(num_images)), None

    # Pick anchor image with highest total inlier connectivity
    anchor_idx = max(accepted_indices, key=lambda i: sum(v[1] for v in adj_matrix[i].values()))
    
    H_global = {anchor_idx: np.eye(3, dtype=np.float64)}
    visited = {anchor_idx}
    queue = [anchor_idx]

    # Best-first traversal to chain homographies along the strongest edges
    while queue:
        curr = queue.pop(0)
        # Sort neighbors by inlier count (highest weight first)
        sorted_neighbors = sorted(adj_matrix[curr].items(), key=lambda x: x[1][1], reverse=True)
        
        for neighbor, (H_curr_to_neigh, _) in sorted_neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                H_neigh_to_curr = np.linalg.inv(H_curr_to_neigh)
                H_global[neighbor] = H_global[curr] @ H_neigh_to_curr
                queue.append(neighbor)

    rejected_indices = [i for i in range(num_images) if i not in visited]
    return H_global, rejected_indices, anchor_idx


def get_canvas_bounds(images, H_global):
    """Calculate canvas size and required translation offset with bounds capping."""
    all_corners = []
    valid_indices = []

    for idx, H in H_global.items():
        h, w = images[idx].shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, H)
        
        # Check if projected corners blow past maximum canvas limit
        if np.abs(warped).max() > MAX_CANVAS_DIM:
            print(f"   [WARNING] Image {idx} excluded due to perspective overflow.")
            continue

        all_corners.append(warped)
        valid_indices.append(idx)

    if not all_corners:
        return None, 0, 0, []

    all_corners = np.vstack(all_corners)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype=np.float64)

    return translation, x_max - x_min, y_max - y_min, valid_indices


def stitch_folder(folder_path):
    """Process a single image folder and return panorama array + metadata."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.bmp')
    image_paths = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.endswith(valid_extensions)
    ])

    if len(image_paths) < 2:
        return None, f"Insufficient images ({len(image_paths)}) found in folder."

    images, kp_list, desc_list, filenames = [], [], [], []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        fname = os.path.basename(path)
        kp, desc = detect_and_compute(img, feature_type=FEATURE_TYPE)
        
        images.append(img)
        kp_list.append(kp)
        desc_list.append(desc)
        filenames.append(fname)

    adj_matrix = build_image_graph(images, kp_list, desc_list)
    H_global, rejected_indices, anchor_idx = filter_and_chain_transforms(adj_matrix, len(images))

    if H_global is None or len(H_global) < 2:
        rejected_logs = [filenames[i] for i in range(len(images))]
        return None, f"Failed to connect images (Rejected all: {rejected_logs})"

    # Calculate Canvas Bounding Box & Sanity Guard
    bounds_result = get_canvas_bounds(images, H_global)
    if bounds_result[0] is None:
        return None, "Failed: Homographies unstable (canvas bounds overflow)."

    T, canvas_w, canvas_h, valid_indices = bounds_result
    panorama = np.full((canvas_h, canvas_w, 3), OUTPUT_CANVAS_COLOR, dtype=np.uint8)

    for idx in valid_indices:
        H_final = T @ H_global[idx]
        warped_img = cv2.warpPerspective(
            images[idx], H_final, (canvas_w, canvas_h), 
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=OUTPUT_CANVAS_COLOR
        )
        mask = (warped_img > 0)
        panorama[mask] = warped_img[mask]

    rejected_logs = [filenames[i] for i in range(len(images)) if i not in valid_indices]
    status_msg = f"Anchor: {filenames[anchor_idx]} | Accepted: {len(valid_indices)}/{len(images)} | Rejected: {rejected_logs}"
    return panorama, status_msg


def process_dataset_batch(dataset_root_path, max_folders=3):
    """Iterate over subfolders inside dataset root directory."""
    subfolders = sorted([
        os.path.join(dataset_root_path, d) for d in os.listdir(dataset_root_path)
        if os.path.isdir(os.path.join(dataset_root_path, d))
    ])[:max_folders]

    print(f"\n=======================================================")
    print(f"BATCH PROCESSING: {dataset_root_path}")
    print(f"Found {len(subfolders)} subfolders to test.")
    print(f"=======================================================")

    out_dir = os.path.join(dataset_root_path, "stitched_results")
    os.makedirs(out_dir, exist_ok=True)

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        print(f"\nProcessing '{folder_name}'...")
        
        panorama, log = stitch_folder(folder)
        print(f"Result -> {log}")

        if panorama is not None:
            save_path = os.path.join(out_dir, f"{folder_name}_panorama.jpg")
            cv2.imwrite(save_path, panorama)
            print(f"Saved: {save_path}")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Process 3 subfolders from VPG dataset
    vpg_path = os.path.join(SCRIPT_DIR, "VPG")
    if os.path.exists(vpg_path):
        process_dataset_batch(vpg_path, max_folders=3)

    # Process 3 subfolders from GES-50 dataset
    ges_path = os.path.join(SCRIPT_DIR, "GES-50")
    if os.path.exists(ges_path):
        process_dataset_batch(ges_path, max_folders=3)