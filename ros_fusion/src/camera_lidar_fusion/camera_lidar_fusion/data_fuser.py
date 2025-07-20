import time
from queue import Queue

import cv2

from camera_lidar_fusion.utils.draw_utils import (
    draw_2d_bounding_boxes,
    draw_3d_bounding_boxes,
)
from camera_lidar_fusion.utils.fusion_utils import (
    average_bounding_boxes,
    match_and_compute_iou,
)
from camera_lidar_fusion.utils.object_desciption_util import (
    get_coco_labels,
    get_color_map,
    get_nuscenes_labels,
)
from camera_lidar_fusion.utils.projection_utils import (
    init_projection_parameters,
    project_3d_bb_to_image,
)


class DataFuser:
    """
    Class to handle the fusion of 2D and 3D object detection results.
    It projects 3D bounding boxes onto the image plane and allows for visualization
    of the fused data, including options for computing averages and displaying depth.
    """

    def __init__(self):
        """
        Initializes the DataFuser with the necessary parameters for projection.
        Initializes camera intrinsics, transformation from lidar to camera,
        and image size based on the NuScenes dataset.

        Args:
            nusc (NuScenes): An instance of the NuScenes dataset.
        """

        (
            self.cam_intrinsics,
            self.T_lidar_to_cam,
            self.image_size,
        ) = init_projection_parameters()

    def project_3d_bb_to_image(self, pc_results):
        """
        Projects 3D bounding boxes from point cloud results onto the image plane.
        This method takes the 3D bounding boxes, labels, and scores from the point cloud results,
        and uses the camera intrinsics and transformation matrix to project them onto the image.
        It returns the projected bounding boxes in a format suitable for visualization.

        Args:
            pc_results: 3D bounding box results from the point cloud detection model.

        Returns:
            projected_boxes: Projected bounding boxes in image coordinates.
        """
        bounding_boxes = pc_results["bboxes_3d"]
        labels = pc_results["labels_3d"]
        scores = pc_results["scores_3d"]

        projected_boxes = project_3d_bb_to_image(
            bounding_boxes,
            labels,
            scores,
            self.cam_intrinsics,
            self.T_lidar_to_cam,
            self.image_size,
        )

        return projected_boxes

    def fuse_data_and_display(self, post_process_queue: Queue):
        """
        Fuses 2D and 3D object detection results and displays them in a window.
        This method continuously retrieves data from the post-processing queue,
        projects 3D bounding boxes onto the image plane, and visualizes the results.
        It supports various modes such as computing averages, showing depth information,
        and toggling between camera-only, lidar-only, or average-only views.
        It also allows for pausing and advancing through the frames in presentation mode.

        Args:
            post_process_queue (Queue): Queue containing post-processed data.
            compute_average (bool): Computes average 3D bounding boxes for matched objects.
            show_depth (bool): Visualizes depth information on image plane.
            presentation_mode (bool): Stops the visualization at the first image
                and does not display FPS metrics.
            camera_only (bool): Only displays 2D bounding boxes from the camera object detection.
            lidar_only (bool): Only displays 3D bounding boxes from the lidar object detection.
            average_only (bool): Only displays average 3D bounding boxes from matched objects.
            show_iou_score (bool): Displays the IoU score for matched objects.
            display_highest_certainty (bool): On matches, display either 2D or 3D bounding box based on precision score
        """

        color_map = get_color_map()
        nuscenes_labels = get_nuscenes_labels()
        coco_labels = get_coco_labels()

        paused = False
        advance_once = False

        start_time = 0

        while True:
            if not paused or advance_once:
                advance_once = False

                data = post_process_queue.get()
                image, img_results, pc_results = data

                start_time = time.time()

                projected_boxes = self.project_3d_bb_to_image(pc_results)

                (
                    matches,
                    matched_lid_indices,
                    matched_cam_indices,
                ) = match_and_compute_iou(
                    projected_boxes,
                    img_results,
                    iou_threshold=0.5,
                )

                unmatched_projected_boxes = [
                    box
                    for idx, box in enumerate(projected_boxes)
                    if idx not in matched_lid_indices
                ]

                unmatched_img_results = [
                    box
                    for idx, box in enumerate(img_results)
                    if idx not in matched_cam_indices
                ]

                average_boxes = average_bounding_boxes(
                    matches, coco_labels, nuscenes_labels
                )

                resulting_2d_boxes = unmatched_img_results

                resulting_3d_boxes = unmatched_projected_boxes + (
                    average_boxes if average_boxes is not None else []
                )

                draw_2d_bounding_boxes(
                    image, color_map, coco_labels, resulting_2d_boxes
                )

                draw_3d_bounding_boxes(
                    image, color_map, nuscenes_labels, resulting_3d_boxes
                )

            cv2.imshow("Fused Data", image)
            key = cv2.waitKey(0 if paused else 1) & 0xFF

            if key == ord("q"):
                print("[DataFuser] Quitting presentation loop.")
                break
            elif key == ord(" "):  # Spacebar to toggle pause
                paused = not paused
                print(f"[DataFuser] {'Paused' if paused else 'Continuing'}...")
            elif key == ord("a") and paused:  # Advance one frame
                advance_once = True
                print("[DataFuser] Advancing one frame...")
                continue  # Skip rest of loop to fetch next frame

            print(
                f"[DataFuser] FPS for data fusion: {1 / (time.time() - start_time):.2f}"
            )

        cv2.destroyAllWindows()
