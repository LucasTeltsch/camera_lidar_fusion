from typing import Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes
from utils.projection_utils import transformation_matrix

"""
Utility class for matching camera and lidar bounding boxes by computing IoU (Intersection over Union).
Contains functions for computing average 3D bounding boxes for matches. 
"""


def compute_2d_iou(boxA, boxB) -> float:
    """
    Compute the Intersection over Union (IoU) of two 2D bounding boxes.
    The boxes are defined by their coordinates in the format [xmin, ymin, xmax, ymax].
    The IoU is calculated as the area of intersection divided by the area of union.
    The function returns 0 if the union area is zero to avoid division by zero.

    Args:
        boxA: list with coordinates [xmin, ymin, xmax, ymax] for the first box
        boxB: list with coordinates [xmin, ymin, xmax, ymax] for the second box

    Returns:
        float: The IoU value between the two boxes. Returns 0.0 if the union area is zero.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def get_2d_bbox_from_corners(corners):
    """
    Get the 2D bounding box from the corners of a 3D bounding box.
    The corners are expected to be in the format of a numpy array with shape (8, 2),
    where each row represents a corner with x and y coordinates.
    The function returns the bounding box in the format [xmin, ymin, xmax, ymax].
    This is useful for matching 3D bounding boxes projected onto a 2D image plane with 2D camera bounding boxes.

    Args:
        corners: numpy array of shape (8, 2) representing the corners of the 3D bounding box.

    Returns:
        list: list containing [xmin, ymin, xmax, ymax] coordinates of the 2D bounding box.
    """
    # corners: (8, 2) numpy array with x, y values
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    return [x_min, y_min, x_max, y_max]


def match_and_compute_iou(
    projected_boxes,
    camera_boxes,
    iou_threshold=0.5,
):
    """
    Match projected 3D bounding boxes with 2D camera bounding boxes and compute IoU.
    This function iterates through the projected 3D bounding boxes and the camera bounding boxes,
    computes the 2D IoU for each pair, and returns a list of matches that meet the IoU threshold.
    Each match contains the IoU value, the projected box details, and the camera box details.

    Args:
        projected_boxes: List of projected 3D bounding boxes in image coordinates.
        camera_boxes: List of 2D bounding boxes.
        iou_threshold (float, optional): A threshhold which determines the validity of matches. Defaults to 0.5.

    Returns:
        list: A list of matches, each containing:
            - "iou": The IoU value between the projected and camera bounding boxes.
            - "proj_box": The projected 3D bounding box details.
            - "cam_box": The camera 2D bounding box details, including bbox coordinates, score, and label.
    """
    matches = []

    matched_lid_indices = set()
    matched_cam_indices = set()

    for i, proj_box in enumerate(projected_boxes):
        proj_corners = proj_box["corners"]
        proj_bbox = get_2d_bbox_from_corners(proj_corners)

        for j, camera_box in enumerate(camera_boxes):
            xmin, ymin, xmax, ymax, score, label = camera_box
            cam_bbox = [xmin, ymin, xmax, ymax]

            iou = compute_2d_iou(proj_bbox, cam_bbox)

            if iou >= iou_threshold:
                matches.append(
                    {
                        "iou": iou,
                        "proj_box": proj_box,
                        "cam_box": {
                            "bbox": cam_bbox,
                            "score": score,
                            "label": label,
                        },
                    }
                )
                matched_lid_indices.add(i)
                matched_cam_indices.add(j)

    return matches, matched_lid_indices, matched_cam_indices


def adjust_projected_corners_to_bbox(corners, target_bbox):
    """
    Adjusts the projected 3D box corners to fit within the targeted 2D bbox.

    corners: (8, 2) np.array, 2D projection of 3D corners.
    target_bbox: [xmin, ymin, xmax, ymax] - the averaged box.

    Returns: adjusted_corners (8, 2)
    """
    corners = np.array(corners)
    original_bbox = get_2d_bbox_from_corners(corners)
    orig_xmin, orig_ymin, orig_xmax, orig_ymax = original_bbox
    target_xmin, target_ymin, target_xmax, target_ymax = target_bbox

    # Normalize corners to 0-1 in original bbox space
    norm_x = (corners[:, 0] - orig_xmin) / max(orig_xmax - orig_xmin, 1e-6)
    norm_y = (corners[:, 1] - orig_ymin) / max(orig_ymax - orig_ymin, 1e-6)

    # Scale to new bbox
    new_x = norm_x * (target_xmax - target_xmin) + target_xmin
    new_y = norm_y * (target_ymax - target_ymin) + target_ymin

    adjusted_corners = np.stack((new_x, new_y), axis=-1)
    return adjusted_corners.astype(int)


def create_labels(
    camera_label,
    camera_score,
    lidar_label,
    lidar_score,
    iou_score,
    show_iou_score=False,
) -> Tuple[str, str]:
    """
    Creates labels for objects that were matched. The color of the matched box is determined by the highest confidence score.
    If both sensors return the same classifictation result, the class and both scores are displayed.
    If they return different classes, both confidence scores and classes are displayed. The more confident result is at the front.

    Args:
        camera_label: Label number corresponding to class from COCO dataset
        camera_score: Detection score of camera object detection
        lidar_label: Label number corresponding to class from NuScenes dataset
        lidar_score: Detection score of lidar object detection
        iou_score: The computed IoU score of the matched boxes
        show_iou_score: Adds the iou_score to the box label

    Returns:
        Tuple[str, str]: Tuple of labels:
            - label: class for determining color of box
            - matchted_label: text for for machted object boxes
    """

    label = camera_label if camera_score > lidar_score else lidar_label

    matched_label = "AVERAGE: "

    if camera_label == lidar_label:
        if camera_score > lidar_score:
            matched_label += (
                f"{camera_label}: {camera_score:.2f} (CAM), {lidar_score:.2f} (LID)"
            )
        else:
            matched_label += (
                f"{lidar_label}: {lidar_score:.2f} (LID), {camera_score:.2f} (CAM)"
            )

    else:
        if camera_score > lidar_score:
            matched_label += f"{camera_label}: {camera_score:.2f} (CAM), {lidar_label}: {lidar_score:.2f} (LID)"
        else:
            matched_label += f"{lidar_label}: {lidar_score:.2f} (LID), {camera_label}: {camera_score:.2f} (CAM)"

    if show_iou_score:
        matched_label += f", IuO Score: {iou_score:.2f}"

    return label, matched_label


def average_bounding_boxes(matches, show_iou_score=False) -> list:
    """
    For each match found by the IoU process an average 3D bounding box is computed.
    A weighted average by the scores and x and y coordinates is computed to create an improved box.
    Returned are appropriate and adjusted 3D bounding boxes.

    Args:
        matches: List of matched 2D and 3D bounding boxes
        show_iou_score: Adds the iou_score to the box label

    Returns:
        list: List of adjusted 3D bounding boxes or None is no matches were found
    """

    adjusted_boxes = []

    for match in matches:
        camera_box = match["cam_box"]["bbox"]
        camera_score = match["cam_box"]["score"]
        camera_label = match["cam_box"]["label"]

        lidar_box = match["proj_box"]["corners"]
        lidar_score = match["proj_box"]["score"]
        lidar_label = match["proj_box"]["label"]

        iou_score = match["iou"]

        box_3d_to_2d = get_2d_bbox_from_corners(lidar_box)

        weighted_average_x_min = (
            camera_box[0] * camera_score + box_3d_to_2d[0] * lidar_score
        ) / (camera_score + lidar_score)
        weighted_average_y_min = (
            camera_box[1] * camera_score + box_3d_to_2d[1] * lidar_score
        ) / (camera_score + lidar_score)
        weighted_average_x_max = (
            camera_box[2] * camera_score + box_3d_to_2d[2] * lidar_score
        ) / (camera_score + lidar_score)
        weighted_average_y_max = (
            camera_box[3] * camera_score + box_3d_to_2d[3] * lidar_score
        ) / (camera_score + lidar_score)

        averaged_bbox = [
            weighted_average_x_min,
            weighted_average_y_min,
            weighted_average_x_max,
            weighted_average_y_max,
        ]

        adjusted_corners = adjust_projected_corners_to_bbox(lidar_box, averaged_bbox)

        label, matched_label = create_labels(
            camera_label,
            camera_score,
            lidar_label,
            lidar_score,
            iou_score,
            show_iou_score,
        )

        adjusted_boxes.append(
            {
                "corners": adjusted_corners,
                "score": None,
                "label": label,
                "matched_label": matched_label,
            }
        )

        return adjusted_boxes


def get_ground_truth_boxes(
    nusc: NuScenes, sample, T_lidar_to_cam, camera_intrinsics, image_size
):
    """Retrieves ground truth boxes from the NuScenes dataset for a given sample.

    Args:
        nusc (NuScenes): An instance of the NuScenes dataset.
        sample: Sample record containing data about the current sample.
        T_lidar_to_cam: Transformation matrix from LiDAR to camera coordinates.
        camera_intrinsics: Camera intrinsic matrix for projecting 3D points to 2D.
        image_size: Tuple containing the width and height of the image.

    Returns:
        list: List of dictionaries, each containing:
            - "corners": numpy array of shape (8, 2) representing the corners of the 2D bounding box.
            - "label": Class name of the object.
            - "score": -1 for ground truth boxes.
    """
    lidar_token = sample["data"]["LIDAR_TOP"]

    sample_data = nusc.get("sample_data", lidar_token)
    calibrated_sensor = nusc.get(
        "calibrated_sensor", sample_data["calibrated_sensor_token"]
    )

    ego_record = nusc.get("ego_pose", sample_data["ego_pose_token"])

    T_lidar_to_ego = transformation_matrix(
        calibrated_sensor["rotation"], calibrated_sensor["translation"]
    )
    T_ego_to_global = transformation_matrix(
        ego_record["rotation"], ego_record["translation"]
    )
    T_lidar_to_global = T_ego_to_global @ T_lidar_to_ego
    T_global_to_lidar = np.linalg.inv(T_lidar_to_global)
    T_global_to_cam = T_lidar_to_cam @ T_global_to_lidar

    ground_truth_box_tokens = sample["anns"]

    projected_boxes = []

    for box_token in ground_truth_box_tokens:
        annotation = nusc.get("sample_annotation", box_token)
        box = nusc.get_box(annotation["token"])

        corners_3d = box.corners().T
        label = box.name
        score = -1

        lidar_points_hom = np.hstack([corners_3d, np.ones((8, 1))])
        cam_points_hom = (T_global_to_cam @ lidar_points_hom.T).T
        cam_points = cam_points_hom[:, :3]

        # Filter points behind the camera
        if np.any(cam_points[:, 2] < 0.1):
            continue

        projected = camera_intrinsics @ cam_points.T
        projected = projected[:2, :] / projected[2, :]
        projected = projected.T  # (8, 2)

        if image_size:
            w, h = image_size
            if np.all(
                (projected[:, 0] < 0)
                | (projected[:, 0] >= w)
                | (projected[:, 1] < 0)
                | (projected[:, 1] >= h)
            ):
                continue

        projected_boxes.append(
            {"corners": projected.astype(int), "label": label, "score": score}
        )

    return projected_boxes


def evaluate_detection(boxes_2d, boxes_3d, gt_boxes, iou_threshold=0.5):
    """
    Evaluates the detection results by comparing 2D and 3D bounding boxes against ground truth boxes.

    Returns:
        Tuple: Tuple containing:
            - avg_iou: Average IoU score of the matched boxes.
            - tp: True Positives count.
            - fp: False Positives count.
            - fn: False Negatives count.
    """

    iou_scores = []

    # Prepare 2D detections (append dummy IoU placeholder)
    new_boxes_2d = [box_2d + [0.0] for box_2d in boxes_2d]

    # Prepare 3D detections
    new_boxes_3d = boxes_3d.copy()
    for box_3d in new_boxes_3d:
        box_3d["gt_score"] = 0.0

    # Collect all valid (IoU >= threshold and class match) detection-GT pairs
    matches = []  # (iou, gt_idx, pred_id, is_3d)

    for gt_idx, gt_box in enumerate(gt_boxes):
        gt_corners = gt_box["corners"]
        gt_bbox = get_2d_bbox_from_corners(gt_corners)
        gt_label = gt_box["label"]

        # Compare with 2D detections
        for pred_idx, box_2d in enumerate(new_boxes_2d):
            xmin, ymin, xmax, ymax, score, pred_label, _ = box_2d
            pred_bbox = [xmin, ymin, xmax, ymax]

            if pred_label in gt_label:
                iou = compute_2d_iou(gt_bbox, pred_bbox)
                if iou >= iou_threshold:
                    matches.append((iou, gt_idx, f"2d_{pred_idx}", False))

        # Compare with 3D detections (via 2D projection)
        for pred_idx, box_3d in enumerate(new_boxes_3d):
            pred_corners = box_3d["corners"]
            pred_bbox = get_2d_bbox_from_corners(pred_corners)
            pred_label = box_3d["label"]

            if pred_label in gt_label:
                iou = compute_2d_iou(gt_bbox, pred_bbox)
                if iou >= iou_threshold:
                    matches.append((iou, gt_idx, f"3d_{pred_idx}", True))

    # Sort by descending IoU
    matches.sort(reverse=True)

    matched_gt = set()
    matched_preds = set()

    for iou, gt_idx, pred_id, is_3d in matches:
        if gt_idx not in matched_gt and pred_id not in matched_preds:
            matched_gt.add(gt_idx)
            matched_preds.add(pred_id)
            iou_scores.append(iou)

    tp = len(matched_preds)
    fp = len(new_boxes_2d) + len(new_boxes_3d) - tp
    fn = len(gt_boxes) - len(matched_gt)

    avg_iou = np.mean(iou_scores) if iou_scores else 0

    return avg_iou, tp, fp, fn


def add_ground_truth_iou(boxes_2d, boxes_3d, gt_boxes):
    """
    Adds IoU scores from ground truth boxes to the 2D and 3D bounding boxes.

    Args:
        boxes_2d: list of 2D bounding boxes
        boxes_3d: list of 3D bounding boxes
        gt_boxes: list of ground truth boxes

    Returns:
        Tuple: Tuple containing:
            - new_boxes_2d: list of 2D bounding boxes with updated GT IoU
            - new_boxes_3d: list of 3D bounding boxes with updated GT IoU
    """

    new_boxes_2d = []

    for box_2d in boxes_2d:
        new_boxes_2d.append(box_2d + [0.0])

    new_boxes_3d = boxes_3d.copy()
    for box_3d in new_boxes_3d:
        box_3d["gt_score"] = 0.0

    for gt_box in gt_boxes:
        gt_corners = gt_box["corners"]
        gt_bbox = get_2d_bbox_from_corners(gt_corners)

        class_name = gt_box["label"]

        for box_2d in new_boxes_2d:
            xmin, ymin, xmax, ymax, score, label, box_iou = box_2d
            cam_bbox = [xmin, ymin, xmax, ymax]

            iou = compute_2d_iou(gt_bbox, cam_bbox)

            same_class = label in class_name

            if same_class and iou > box_iou:
                box_2d[6] = iou

        for box_3d in new_boxes_3d:
            corners_3d = box_3d["corners"]
            box_2d = get_2d_bbox_from_corners(corners_3d)

            iou = compute_2d_iou(gt_bbox, box_2d)

            same_class = box_3d["label"] in class_name

            if same_class and iou > box_3d["gt_score"]:
                box_3d["gt_score"] = iou

    return new_boxes_2d, new_boxes_3d


def preprocess_labels(img_results, pc_results, coco_labels, nuscenes_labels):

    new_img_results = []

    for img_result in img_results:
        img_result = img_result.tolist()
        img_result[5] = coco_labels[img_result[5]]

        new_img_results.append(img_result)

    labels = []

    for label in pc_results["labels_3d"]:
        labels.append(nuscenes_labels[label])

    pc_results["labels_3d"] = labels

    return new_img_results, pc_results
