import cv2

"""
Utility class for drawing 2D and 3D bounding boxes on an image.
"""


def draw_lines(image, corners, color):
    """
    Draws lines between the corners of a 3D bounding box on the image.
    This function connects the corners of a 3D bounding box to visualize it in 2D space.

    Args:
        image: Image on which to draw the bounding box.
        corners: numpy array of shape (8, 3) representing the corners of the 3D bounding box.
        color: Color to use for drawing the lines.
    """
    corners = corners.astype(int)
    # Define the 12 edges of a 3D box
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # bottom square
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # top square
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # vertical lines
    ]
    for start, end in edges:
        pt1 = tuple(corners[start])
        pt2 = tuple(corners[end])
        cv2.line(image, pt1, pt2, color, 2)


def draw_3d_bounding_boxes(image, color_map, classes, projected_boxes):
    """
    Draws 3D bounding boxes on the image.
    This function iterates through the projected 3D bounding boxes and draws them on the image.

    Args:
        image: Image on which to draw the bounding boxes.
        color_map: Dictionary mapping class names to colors.
        classes: List of class names corresponding to the labels in the projected boxes.
        projected_boxes: List of dictionaries, each containing:
            - "corners": numpy array of shape (8, 3) representing the corners of the 3D bounding box.
            - "label": Index of the class label.
            - "score": Confidence score of the detection (can be None for average boxes).
            - "matched_label": Label for average boxes, if applicable.
    """
    for box in projected_boxes:
        label = box["label"]
        score = box["score"]

        if score is None:
            # If score is None, it means this box is an average box
            label_text = box["matched_label"]
            color = color_map[label]
            
            if "gt_score" in box:
                label_text += f": {round(box['gt_score'], 3)} (gt_iou)"
        else:
            if score == -1:
                color = color_map["ground_truth"]
                label_text = "Ground Truth: " + box["label"]
            else:
                class_name = box["label"]
                color = color_map[class_name]
                label_text = f"{class_name}: {round(box['score'], 3)} (LID)"
        
            if "gt_score" in box:
                class_name = box["label"]
                label_text = f"{class_name}: {round(box['gt_score'], 3)} (gt_iou)"

        draw_lines(image, box["corners"], color)

        top_corner = box["corners"][4]
        pt = (int(top_corner[0]), int(top_corner[1] - 10))  # move label above box
        cv2.putText(
            image,
            label_text,
            pt,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def draw_2d_bounding_boxes(image, color_map, classes, results):
    """
    Draws 2D bounding boxes on the image.
    This function iterates through the results and draws bounding boxes with labels on the image.

    Args:
        image: Image on which to draw the bounding boxes.
        color_map: Dictionary mapping class names to colors.
        classes: List of class names corresponding to the labels in the projected boxes.
        results: List of image object detection results, each containing:
            - xmin: Minimum x-coordinate of the bounding box.
            - ymin: Minimum y-coordinate of the bounding box.
            - xmax: Maximum x-coordinate of the bounding box.
            - ymax: Maximum y-coordinate of the bounding box.
            - score: Confidence score of the detection.
            - label: Index of the class label.
    """
    for result in results:
        if len(result) == 6:
            xmin, ymin, xmax, ymax, score, label = result
        elif len(result) == 7:
            xmin, ymin, xmax, ymax, score, label, gt_iou = result

        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

        class_name = label

        color = color_map[class_name]  # Get color for this class

        # Draw bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

        if len(result) == 6:
            label_text = f"{class_name}: {round(score, 3)} (CAM)"
        elif len(result) == 7:
            label_text = f"{class_name}: {round(gt_iou, 3)} (gt_iou)"

        cv2.putText(
            image,
            label_text,
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
