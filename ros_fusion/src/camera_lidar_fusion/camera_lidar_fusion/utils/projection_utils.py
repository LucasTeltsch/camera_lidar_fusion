import cv2
import numpy as np
from pyquaternion import Quaternion

"""
Utilty class for projecting 3D bounding boxes onto an image.
"""


def transformation_matrix(rotation_matrix, translation_matrix):
    """Creates a 4x4 transformation matrix from rotation and translation.

    Args:
        rotation_matrix : Rotation matrix from camera or lidar calibration.
        translation_matrix : Translation vector from camera or lidar calibration.

    Returns:
        NDArray[float64]: 4x4 transformation matrix combining rotation and translation.
    """
    T = np.eye(4)
    R = Quaternion(rotation_matrix).rotation_matrix
    T[:3, :3] = R
    T[:3, 3] = translation_matrix
    return T


def init_projection_parameters():
    """
    Initialize projection parameters for camera and lidar data.
    This function retrieves the camera and lidar calibration data from the NuScenes dataset,
    computes the transformation matrix from lidar to camera, and returns the camera intrinsics,
    the transformation matrix, and the image size.
    Parameters stay constant for all samples in the dataset.

    Args:
        nusc (NuScenes): An instance of the NuScenes dataset.

    Returns:
        Tuple containing:
            - cam_intrinsics (np.ndarray): Camera intrinsic parameters.
            - T_lidar_to_cam (np.ndarray): Transformation matrix from lidar to camera.
            - image_size (tuple): Size of the camera image in pixels (width, height).
    """

    cam_translation = np.array([0, 1.168, -1.940])
    cam_rotation = np.array(
        [1.2091995761561452, -1.2091995761561452, 1.209199576156145]
    )

    cam_rotation, _ = cv2.Rodrigues(cam_rotation)

    T_cam = np.eye(4)
    T_cam[:3, :3] = cam_rotation
    T_cam[:3, 3] = cam_translation

    fx = 2688.4255794073424
    fy = 2702.8870683505797
    cx = 2028.8346149531565
    cy = 1101.3150697657088
    cam_intrinsics = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    lidar_translation = [
        0,
        1.613,
        -1.265,
    ]  # If its algined with camera coordinate system
    T_lidar = np.eye(4)  # Use the identiy as rotation matrix
    T_lidar[:3, 3] = lidar_translation

    T_lidar_to_cam = np.linalg.inv(T_cam) @ T_lidar

    image_size = (4112 / 4, 2176 / 4)
    return cam_intrinsics, T_lidar_to_cam, image_size


def get_3d_box_corners(x, y, z, dx, dy, dz, yaw):
    """
    Computes the 3D bounding box corners given the center coordinates, dimensions, and yaw angle.

    Args:
        x: x-coordinate of the bounding box center.
        y: y-coordinate of the bounding box center.
        z: z-coordinate of the bounding box center.
        dx: width of the bounding box.
        dy: height of the bounding box.
        dz: depth of the bounding box.
        yaw: yaw angle of the bounding box in radians.

    Returns:
        NDArray: A numpy array of shape (8, 3) containing the coordinates of the 8 corners of the bounding box.
    """
    corners = np.array(
        [
            [dx / 2, dy / 2, 0],
            [-dx / 2, dy / 2, 0],
            [-dx / 2, -dy / 2, 0],
            [dx / 2, -dy / 2, 0],
            [dx / 2, dy / 2, dz],
            [-dx / 2, dy / 2, dz],
            [-dx / 2, -dy / 2, dz],
            [dx / 2, -dy / 2, dz],
        ]
    )  # shape (8, 3)

    # Rotation matrix around Z-axis
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

    # Rotate and translate corners
    corners = (R @ corners.T).T
    corners += np.array([x, y, z])
    return corners  # shape (8, 3)


def project_3d_bb_to_image(
    bounding_boxes, labels, scores, camera_intrinsics, T_lidar_to_cam, image_size=None
):
    """
    Projects 3D bounding boxes from LiDAR space to 2D image space using camera intrinsics and
    extrinsics (LiDAR-to-camera transformation). Boxes with low confidence or irrelevant classes
    are filtered out.

    Args:
        bounding_boxes (np.ndarray): An array of shape (N, 7) where each row represents a 3D
            bounding box in LiDAR coordinates with format (x, y, z, dx, dy, dz, yaw).
        labels (np.ndarray or list of int): List of integer class labels corresponding to each
            bounding box.
        scores (np.ndarray or list of float): Confidence scores associated with each bounding box.
        camera_intrinsics (np.ndarray): A 3x3 camera intrinsic matrix.
        T_lidar_to_cam (np.ndarray): A 4x4 transformation matrix converting points from LiDAR
            coordinates to camera coordinates.
        image_size (tuple, optional): Tuple (width, height) specifying the size of the image.
            If provided, boxes projected entirely outside the image are discarded. Defaults to None.

    Returns:
        List[dict]: A list of dictionaries, each containing:
            - "corners" (np.ndarray): An (8, 2) array of projected 2D corner points.
            - "label" (int): The class label of the object.
            - "score" (float): The confidence score of the detection.
    """
    projected_boxes = []

    for bbox, label, score in zip(bounding_boxes, labels, scores):
        if score < 0.3:
            continue
        # Skip Traffic Cone and Barrier classes
        if label == 8 or label == 9:
            continue

        x, y, z, dx, dy, dz, yaw = bbox[:7]
        corners_3d = get_3d_box_corners(x, y, z, dx, dy, dz, yaw)  # (8, 3)
        lidar_points_hom = np.hstack([corners_3d, np.ones((8, 1))])
        cam_points_hom = (T_lidar_to_cam @ lidar_points_hom.T).T
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
