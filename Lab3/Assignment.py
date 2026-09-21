import os
import glob
import cv2
import numpy as np

# =====================================================================
# CONFIGURATION & HYPERPARAMETERS
# =====================================================================
FEATURE_TYPE = 'SIFT'        # 'SIFT' or 'ORB'
MAX_FEATURES = 3000          # Keypoint budget per image
RATIO_TEST_THRESH = 0.75     # Lowe's ratio test threshold
RANSAC_REPROJ_THRESH = 4.0   # Reprojection threshold in pixels
MIN_INLIER_COUNT = 15        # Minimum matches to form a graph edge

MAX_CANVAS_DIM = 18000       # Allows wide panoramas like S02 without memory crash
OUTPUT_CANVAS_COLOR = (0, 0, 0)
# =====================================================================


def detect_and_compute(image, feature_type='SIFT'):
    """Extract keypoints and descriptors from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if feature_type.upper() == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=MAX_FEATURES)
    elif feature_type.upper() == 'ORB':
        detector = cv2.ORB_create(nfeatures=MAX_FEATURES)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2, feature_type='SIFT', ratio_thresh=0.75):
    """Perform KNN descriptor matching and apply ratio test."""
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


def is_valid_homography(H):
    """Sanity check to reject mathematical singularities."""
    if H is None or H.shape != (3, 3):
        return False

    if abs(H[2, 2]) < 1e-8:
        return False
        
    H_norm = H / H[2, 2]
    det = np.linalg.det(H_norm[:2, :2])

    # Reject collapsed planes or extreme inversions
    if det < 0.005 or det > 200.0:
        return False

    return True


def estimate_homography_ransac(kp1, kp2, matches, reproj_thresh=4.0):
    """Estimate Homography using RANSAC."""
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
    """Compute pairwise homographies across all image combinations."""
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

            if inliers >= MIN_INLIER_COUNT and is_valid_homography(H):
                adj_matrix[i][j] = (H, inliers)
                adj_matrix[j][i] = (np.linalg.inv(H), inliers)

    return adj_matrix


def filter_and_chain_transforms(adj_matrix, num_images):
    """Chain pairwise homographies using BFS starting from anchor node."""
    accepted_indices = [i for i in adj_matrix if len(adj_matrix[i]) > 0]
    if not accepted_indices:
        return None, list(range(num_images)), None

    # Pick anchor image with highest total inlier connectivity
    anchor_idx = max(accepted_indices, key=lambda i: sum(v[1] for v in adj_matrix[i].values()))
    
    H_global = {anchor_idx: np.eye(3, dtype=np.float64)}
    visited = {anchor_idx}
    queue = [anchor_idx]

    while queue:
        curr = queue.pop(0)
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
    """
    Project image boundaries onto reference space.
    Filters out individual outlier images if their transform exceeds MAX_CANVAS_DIM.
    """
    valid_corners = []
    valid_indices = []
    overflow_indices = []

    for idx, H in H_global.items():
        h, w = images[idx].shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, H)
        
        x_min_i, y_min_i = warped.min(axis=0).ravel()
        x_max_i, y_max_i = warped.max(axis=0).ravel()
        
        # Check if individual image exceeds max bounds
        if (x_max_i - x_min_i) > MAX_CANVAS_DIM or (y_max_i - y_min_i) > MAX_CANVAS_DIM:
            overflow_indices.append(idx)
            continue

        valid_corners.append(warped)
        valid_indices.append(idx)

    if not valid_corners:
        return None, 0, 0, [], list(H_global.keys())

    all_corners = np.vstack(valid_corners)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    # Final Overall Canvas Guard
    if canvas_w > MAX_CANVAS_DIM or canvas_h > MAX_CANVAS_DIM:
        return None, 0, 0, [], list(H_global.keys())

    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype=np.float64)

    return translation, canvas_w, canvas_h, valid_indices, overflow_indices


def stitch_folder(folder_path):
    """Execution pipeline for an image folder."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.bmp')
    image_paths = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.endswith(valid_extensions)
    ])

    if len(image_paths) < 2:
        return None, f"Insufficient images ({len(image_paths)}) found."

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
    H_global, initial_rejected, anchor_idx = filter_and_chain_transforms(adj_matrix, len(images))

    if H_global is None or len(H_global) < 2:
        rejected_filenames = [filenames[i] for i in range(len(images))]
        return None, f"Failed: Insufficient matches across images. (Rejected: {rejected_filenames})"

    T, canvas_w, canvas_h, valid_indices, overflow_indices = get_canvas_bounds(images, H_global)
    
    total_rejected_indices = list(set(initial_rejected + overflow_indices))
    
    if T is None or len(valid_indices) < 2:
        rejected_filenames = [filenames[i] for i in range(len(images))]
        return None, f"Skipped: Perspective drift exceeds canvas bounds. (Rejected: {rejected_filenames})"

    panorama = np.full((canvas_h, canvas_w, 3), OUTPUT_CANVAS_COLOR, dtype=np.uint8)

    # Render accepted warped images onto the canvas
    for idx in valid_indices:
        H_final = T @ H_global[idx]
        warped_img = cv2.warpPerspective(
            images[idx], H_final, (canvas_w, canvas_h), 
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=OUTPUT_CANVAS_COLOR
        )
        mask = (warped_img > 0)
        panorama[mask] = warped_img[mask]

    rejected_logs = [filenames[i] for i in total_rejected_indices]
    status_msg = f"Anchor: {filenames[anchor_idx]} | Accepted: {len(valid_indices)}/{len(images)} | Rejected: {rejected_logs}"
    return panorama, status_msg


def process_dataset_batch(dataset_root_path, max_folders=3):
    """Run batch evaluation across dataset folders."""
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

    vpg_path = os.path.join(SCRIPT_DIR, "VPG")
    if os.path.exists(vpg_path):
        process_dataset_batch(vpg_path, max_folders=3)

    ges_path = os.path.join(SCRIPT_DIR, "GES-50")
    if os.path.exists(ges_path):
        process_dataset_batch(ges_path, max_folders=3)