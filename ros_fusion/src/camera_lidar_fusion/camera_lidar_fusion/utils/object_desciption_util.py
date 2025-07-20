import random
from typing import Dict, Tuple

"""
Generates a unified set of labels and color mapping for COCO and NuScenes datasets.
This module provides functions to retrieve labels from both datasets and create a color map for visualization.
"""


def get_nuscenes_labels() -> Dict[int, str]:
    """
    Generates a dictionary of NuScenes labels with their corresponding integer IDs.

    Returns:
        Dict[int, str]: A dictionary where keys are integer IDs and values are the corresponding NuScenes class names.
    """

    return {
        0: "car",
        1: "truck",
        2: "trailer",
        3: "bus",
        4: "construction_vehicle",
        5: "bicycle",
        6: "motorcycle",
        7: "pedestrian",
        8: "traffic_cone",
        9: "barrier",
    }


def get_coco_labels() -> Dict[int, str]:
    """
    Generates a dictionary of COCO labels with their corresponding integer IDs.

    Returns:
        Dict[int, str]: A dictionary where keys are integer IDs and values are the corresponding COCO class names.
    """

    return {
        0: "pedestrian",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
    }


def get_color_map() -> Dict[str, Tuple[int, int, int]]:
    """
    Generates a color map for the unified set of labels from COCO and NuScenes datasets.
    This function combines the labels from both datasets, removes duplicates, and assigns a random color to each unique label.
    The color mapping is consistent across runs due to a fixed random seed.

    Returns:
        Dict[str, tuple]: A dictionary where keys are class names and values are tuples representing RGB colors.
    """

    coco_values = list(get_coco_labels().values())
    nuscenes_values = list(get_nuscenes_labels().values())
    unified_values = list(set(coco_values + nuscenes_values))

    # For consistent color mapping
    random.seed(48)
    unified_values.sort()

    color_map = {}
    for label in unified_values:
        color_map[label] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
    return color_map
