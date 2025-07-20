from typing import Tuple

import numpy as np

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
            xmin, ymin, xmax, ymax, score, label = camera_box.tolist()
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
                            "label": int(label),
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
    coco_labels,
    camera_label,
    camera_score,
    nuscenes_labels,
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
        coco_labels: COCO labels for determining class names for camera object detection
        camera_label: Label number corresponding to class from COCO dataset
        camera_score: Detection score of camera object detection
        nuscenes_labels: NuScenes labels for determining class names for lidar object detection
        lidar_label: Label number corresponding to class from NuScenes dataset
        lidar_score: Detection score of lidar object detection
        iou_score: The computed IoU score of the matched boxes
        show_iou_score: Adds the iou_score to the box label

    Returns:
        Tuple[str, str]: Tuple of labels:
            - label: class for determining color of box
            - matchted_label: text for for machted object boxes
    """

    label = (
        coco_labels[camera_label]
        if camera_score > lidar_score
        else nuscenes_labels[lidar_label]
    )

    matched_label = "AVERAGE: "

    if coco_labels[camera_label] == nuscenes_labels[lidar_label]:
        if camera_score > lidar_score:
            matched_label += f"{coco_labels[camera_label]}: {camera_score:.2f} (CAM), {lidar_score:.2f} (LID)"
        else:
            matched_label += f"{nuscenes_labels[lidar_label]}: {lidar_score:.2f} (LID), {camera_score:.2f} (CAM)"

    else:
        if camera_score > lidar_score:
            matched_label += f"{coco_labels[camera_label]}: {camera_score:.2f} (CAM), {nuscenes_labels[lidar_label]}: {lidar_score:.2f} (LID)"
        else:
            matched_label += f"{nuscenes_labels[lidar_label]}: {lidar_score:.2f} (LID), {coco_labels[camera_label]}: {camera_score:.2f} (CAM)"

    if show_iou_score:
        matched_label += f", IuO Score: {iou_score:.2f}"

    return label, matched_label


def average_bounding_boxes(
    matches, coco_labels, nuscenes_labels, show_iou_score=False
) -> list:
    """
    For each match found by the IoU process an average 3D bounding box is computed.
    A weighted average by the scores and x and y coordinates is computed to create an improved box.
    Returned are appropriate and adjusted 3D bounding boxes.

    Args:
        matches: List of matched 2D and 3D bounding boxes
        coco_labels: COCO labels for determining class names for camera object detection
        nuscenes_labels: NuScenes labels for determining class names for lidar object detection
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
            coco_labels,
            camera_label,
            camera_score,
            nuscenes_labels,
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
