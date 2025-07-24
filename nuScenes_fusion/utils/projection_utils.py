import os
import os.path as osp
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from nuscenes.lidarseg.lidarseg_utils import paint_points_label
from nuscenes.nuscenes import NuScenes
from nuscenes.panoptic.panoptic_utils import paint_panop_points_label
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

"""
Utilty class for projecting 3D bounding boxes onto an image.
"""


def get_calibration(nusc: NuScenes, token):
    """
    Retrieve camera calibration data from the NuScenes dataset.
    This function retrieves the translation, rotation, and camera intrinsic parameters
    for a given sample data token in the NuScenes dataset.

    Args:
        nusc (NuScenes): An instance of the NuScenes dataset.
        token: Token for camera or lidar sample data.

    Returns:
        tuple: A tuple containing:
            - translation (np.ndarray): Translation vector of the camera or lidar.
            - rotation (np.ndarray): Rotation matrix of the camera or lidar.
            - camera_intrinsic (np.ndarray): Camera intrinsic parameters.
    """

    sample_data = nusc.get("sample_data", token)
    calibrated_sensor = nusc.get(
        "calibrated_sensor", sample_data["calibrated_sensor_token"]
    )

    translation = calibrated_sensor["translation"]
    rotation = calibrated_sensor["rotation"]
    camera_intrinsic = calibrated_sensor["camera_intrinsic"]

    return translation, rotation, camera_intrinsic


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


def init_projection_parameters(nusc: NuScenes):
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
    sample = nusc.sample[0]
    camera_token = sample["data"]["CAM_FRONT"]
    lidar_token = sample["data"]["LIDAR_TOP"]

    camera_data = nusc.get("sample_data", camera_token)
    camera_path = nusc.dataroot + "/" + camera_data["filename"]

    image = cv2.imread(camera_path)

    h, w = image.shape[:2]
    image_size = (w, h)

    cam_translation, cam_rotation, cam_intrinsics = get_calibration(nusc, camera_token)
    lidar_translation, lidar_rotation, _ = get_calibration(nusc, lidar_token)

    T_cam = transformation_matrix(cam_rotation, cam_translation)  # Camera pose
    T_lidar = transformation_matrix(lidar_rotation, lidar_translation)  # Lidar pose
    T_lidar_to_cam = np.linalg.inv(T_cam) @ T_lidar

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
        if label == 'traffic_cone' or label == 'barrier':
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


def draw_points_on_image_cv2(im, points, coloring, dot_size=2, colormap=plt.cm.jet):
    """
    Draws 2D points onto a BGR image (as loaded with cv2.imread) using colors determined by
    a scalar `coloring` array and a specified colormap.

    Args:
        im (np.ndarray): Input image in BGR format as returned by cv2.imread.
        points (np.ndarray): Array of shape (2, N) representing 2D point coordinates (x, y).
        coloring (np.ndarray): Array of shape (N,) with scalar values for each point, used to
            determine the color via the colormap.
        dot_size (int, optional): Radius of each drawn dot in pixels. Defaults to 2.
        colormap (matplotlib.colors.Colormap, optional): A matplotlib colormap used to map
            scalar `coloring` values to RGB colors. Defaults to `plt.cm.jet`.

    Returns:
        np.ndarray: Output image in BGR format with points drawn on it.
    """

    im = im.copy()
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    # Normalize coloring to [0, 1] for colormap
    coloring_norm = (coloring - np.min(coloring)) / (
        np.max(coloring) - np.min(coloring)
    )
    colors_rgb = colormap(coloring_norm)[:, :3]  # shape (N, 3), values in [0, 1]

    for i in range(points.shape[1]):
        x = int(points[0, i])
        y = int(points[1, i])

        rgb = colors_rgb[i]
        bgr = tuple(int(255 * c) for c in rgb[::-1])  # Convert RGB to BGR

        cv2.circle(im, (x, y), dot_size, bgr, thickness=-1)

    # Convert back to RGB before returning, to preserve original format
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


###
# Following functions adopted from from nuscenes.nuscenes import NuScenes
# and adjusted to serve the purpose of rendering depth information to an
# image. In principle, the loading of the image and point cloud was adapted.
# Additionally the drawing of the points was customized so that it is
# compatible with OpenCV.
# ###


def render_pointcloud_in_image(
    nusc: NuScenes,
    sample_token: str,
    img_path,
    pc_path,
    dot_size: int = 5,
    pointsensor_channel: str = "LIDAR_TOP",
    camera_channel: str = "CAM_FRONT",
    out_path: str = None,
    render_intensity: bool = False,
    show_lidarseg: bool = False,
    filter_lidarseg_labels: List = None,
    ax: Axes = None,
    show_lidarseg_legend: bool = False,
    verbose: bool = True,
    lidarseg_preds_bin_path: str = None,
    show_panoptic: bool = False,
):
    """
    Scatter-plots a pointcloud on top of image.
    :param sample_token: Sample token.
    :param dot_size: Scatter plot dot size.
    :param pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_TOP'.
    :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
    :param out_path: Optional path to save the rendered figure to disk.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidarseg labels instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
    :param ax: Axes onto which to render.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param verbose: Whether to display the image in a window.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    if show_lidarseg:
        show_panoptic = False
    sample_record = nusc.get("sample", sample_token)

    # Here we just grab the front camera and the point sensor.
    pointsensor_token = sample_record["data"][pointsensor_channel]
    camera_token = sample_record["data"][camera_channel]

    points, coloring, im = map_pointcloud_to_image(
        nusc,
        img_path,
        pc_path,
        pointsensor_token,
        camera_token,
        render_intensity=render_intensity,
        show_lidarseg=show_lidarseg,
        filter_lidarseg_labels=filter_lidarseg_labels,
        lidarseg_preds_bin_path=lidarseg_preds_bin_path,
        show_panoptic=show_panoptic,
    )

    colored_image = draw_points_on_image_cv2(
        im, points, coloring, dot_size=2, colormap=plt.cm.jet
    )

    return colored_image


def map_pointcloud_to_image(
    nusc: NuScenes,
    img_path,
    pc_path,
    pointsensor_token: str,
    camera_token: str,
    min_dist: float = 1.0,
    render_intensity: bool = False,
    show_lidarseg: bool = False,
    filter_lidarseg_labels: List = None,
    lidarseg_preds_bin_path: str = None,
    show_panoptic: bool = False,
) -> Tuple:
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidar intensity instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """

    cam = nusc.get("sample_data", camera_token)
    pointsensor = nusc.get("sample_data", pointsensor_token)

    im = cv2.imread(img_path)
    pc = LidarPointCloud.from_file(pc_path)

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_record["translation"]))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get("ego_pose", pointsensor["ego_pose_token"])
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
    pc.translate(np.array(poserecord["translation"]))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get("ego_pose", cam["ego_pose_token"])
    pc.translate(-np.array(poserecord["translation"]))
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
    pc.translate(-np.array(cs_record["translation"]))
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    if render_intensity:
        assert pointsensor["sensor_modality"] == "lidar", (
            "Error: Can only render intensity for lidar, "
            "not %s!" % pointsensor["sensor_modality"]
        )
        # Retrieve the color from the intensities.
        # Performs arbitary scaling to achieve more visually pleasing results.
        intensities = pc.points[3, :]
        intensities = (intensities - np.min(intensities)) / (
            np.max(intensities) - np.min(intensities)
        )
        intensities = intensities**0.1
        intensities = np.maximum(0, intensities - 0.5)
        coloring = intensities
    elif show_lidarseg or show_panoptic:
        assert pointsensor["sensor_modality"] == "lidar", (
            "Error: Can only render lidarseg labels for lidar, "
            "not %s!" % pointsensor["sensor_modality"]
        )

        gt_from = "lidarseg" if show_lidarseg else "panoptic"
        semantic_table = getattr(nusc, gt_from)

        if lidarseg_preds_bin_path:
            sample_token = nusc.get("sample_data", pointsensor_token)["sample_token"]
            lidarseg_labels_filename = lidarseg_preds_bin_path
            assert os.path.exists(lidarseg_labels_filename), (
                "Error: Unable to find {} to load the predictions for sample token {} (lidar "
                "sample data token {}) from.".format(
                    lidarseg_labels_filename, sample_token, pointsensor_token
                )
            )
        else:
            if (
                len(semantic_table) > 0
            ):  # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
                lidarseg_labels_filename = osp.join(
                    nusc.dataroot,
                    nusc.get(gt_from, pointsensor_token)["filename"],
                )
            else:
                lidarseg_labels_filename = None

        if lidarseg_labels_filename:
            # Paint each label in the pointcloud with a RGBA value.
            if show_lidarseg:
                coloring = paint_points_label(
                    lidarseg_labels_filename,
                    filter_lidarseg_labels,
                    nusc.lidarseg_name2idx_mapping,
                    nusc.colormap,
                )
            else:
                coloring = paint_panop_points_label(
                    lidarseg_labels_filename,
                    filter_lidarseg_labels,
                    nusc.lidarseg_name2idx_mapping,
                    nusc.colormap,
                )

        else:
            coloring = depths
            print(
                f"Warning: There are no lidarseg labels in {nusc.version}. Points will be colored according "
                f"to distance from the ego vehicle instead."
            )
    else:
        # Retrieve the color from the depth.
        coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(
        pc.points[:3, :], np.array(cs_record["camera_intrinsic"]), normalize=True
    )

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.

    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[0] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring, im
