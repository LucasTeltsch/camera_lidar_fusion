import time
from queue import Queue

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from utils.draw_utils import draw_2d_bounding_boxes, draw_3d_bounding_boxes
from utils.fusion_utils import (
    add_ground_truth_iou,
    average_bounding_boxes,
    evaluate_detection,
    get_ground_truth_boxes,
    match_and_compute_iou,
    preprocess_labels,
)
from utils.object_desciption_util import (
    get_coco_labels,
    get_color_map,
    get_nuscenes_labels,
)
from utils.projection_utils import (
    init_projection_parameters,
    project_3d_bb_to_image,
    render_pointcloud_in_image,
)


class DataFuser:
    """
    Class to handle the fusion of 2D and 3D object detection results.
    It projects 3D bounding boxes onto the image plane and allows for visualization
    of the fused data, including options for computing averages and displaying depth.
    """

    def __init__(self, nusc: NuScenes):
        """
        Initializes the DataFuser with the necessary parameters for projection.
        Initializes camera intrinsics, transformation from lidar to camera,
        and image size based on the NuScenes dataset.

        Args:
            nusc (NuScenes): An instance of the NuScenes dataset.
        """
        self.nusc = nusc
        (
            self.cam_intrinsics,
            self.T_lidar_to_cam,
            self.image_size,
        ) = init_projection_parameters(nusc)

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

    def fuse_data_and_display(
        self,
        nusc,
        post_process_queue: Queue,
        compute_average=False,
        show_depth=False,
        presentation_mode=False,
        camera_only=False,
        lidar_only=False,
        average_only=False,
        show_iou_score=False,
        display_highest_certainty=False,
        ground_truth=False,
        draw_gt_boxes=False,
    ):
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

        global_start_time = time.time()
        processed_frames = 0

        if ground_truth:
            avg_iou_before = []
            avg_iou_after = []
            avg_tp = []
            avg_fp = []
            avg_fn = []

        while True:
            if not paused or advance_once:
                advance_once = False

                data = post_process_queue.get()
                img_path, pc_path, img_results, pc_results, sample = data

                if data == (None, None, None, None, None):
                    post_process_queue.task_done()
                    print("[DataFuser] Stopping. No more data available.")
                    break

                img_results, pc_results = preprocess_labels(
                    img_results, pc_results, coco_labels, nuscenes_labels
                )

                if not presentation_mode:
                    start_time = time.time()

                if show_depth:
                    image = render_pointcloud_in_image(
                        self.nusc, sample["token"], img_path, pc_path
                    )
                else:
                    image = cv2.imread(img_path)

                projected_boxes = self.project_3d_bb_to_image(pc_results)

                if ground_truth:
                    gt_boxes = get_ground_truth_boxes(
                        nusc,
                        sample,
                        self.T_lidar_to_cam,
                        self.cam_intrinsics,
                        self.image_size,
                    )

                    if draw_gt_boxes:
                        draw_3d_bounding_boxes(
                            image, color_map, nuscenes_labels, gt_boxes
                        )

                    avg_iou, _, _, _ = evaluate_detection(
                        img_results, projected_boxes, gt_boxes
                    )

                    avg_iou_before.append(avg_iou)

                    print(
                        f"Average ground truth IoU score for 3d boxes BEFORE fusion: {avg_iou:.2f}"
                    )

                if compute_average or average_only or display_highest_certainty:
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

                    average_boxes = average_bounding_boxes(matches, show_iou_score)

                    if display_highest_certainty:
                        highest_certainty_cam_boxes = []
                        highest_certainty_lid_boxes = []
                        for match in matches:
                            camera_box = match["cam_box"]["bbox"]
                            camera_score = match["cam_box"]["score"]
                            camera_label = match["cam_box"]["label"]

                            lidar_box = match["proj_box"]["corners"]
                            lidar_score = match["proj_box"]["score"]
                            lidar_label = match["proj_box"]["label"]

                            if lidar_score >= camera_score:
                                highest_certainty_lid_boxes.append(
                                    {
                                        "corners": lidar_box,
                                        "score": lidar_score,
                                        "label": lidar_label,
                                    }
                                )
                            else:
                                xmin, ymin, xmax, ymax = camera_box
                                highest_certainty_cam_boxes.append(
                                    [xmin, ymin, xmax, ymax, camera_score, camera_label]
                                )
                        resulting_2d_boxes = (
                            unmatched_img_results + highest_certainty_cam_boxes
                        )
                        resulting_3d_boxes = (
                            unmatched_projected_boxes + highest_certainty_lid_boxes
                        )
                    else:
                        resulting_2d_boxes = unmatched_img_results

                        if average_only:
                            resulting_3d_boxes = (
                                average_boxes if average_boxes is not None else []
                            )
                        else:
                            resulting_3d_boxes = unmatched_projected_boxes + (
                                average_boxes if average_boxes is not None else []
                            )

                            if ground_truth:
                                avg_iou, tp, fp, fn = evaluate_detection(
                                    resulting_2d_boxes, resulting_3d_boxes, gt_boxes
                                )

                                avg_iou_after.append(avg_iou)
                                avg_tp.append(tp)
                                avg_fp.append(fp)
                                avg_fn.append(fn)

                                print(
                                    f"Average ground truth IoU score for 3d boxes AFTER fusion: {avg_iou:.2f}"
                                )
                                print("Fused 2D/3D Detection Results:")
                                print(f"  True Positives:  {tp}")
                                print(f"  False Positives: {fp}")
                                print(f"  False Negatives: {fn}")

                else:
                    resulting_2d_boxes = img_results
                    resulting_3d_boxes = projected_boxes

                if ground_truth:
                    resulting_2d_boxes, resulting_3d_boxes = add_ground_truth_iou(
                        resulting_2d_boxes, resulting_3d_boxes, gt_boxes
                    )

                if camera_only or (not lidar_only and not average_only):
                    draw_2d_bounding_boxes(
                        image, color_map, coco_labels, resulting_2d_boxes
                    )

                if lidar_only or average_only or not camera_only:
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

            if presentation_mode:
                paused = True
            else:
                print(
                    f"[DataFuser] FPS for data fusion: {1 / (time.time() - start_time):.2f}"
                )

            processed_frames += 1

        print(
            f"[DataFuser] Average FPS for data fusion: {processed_frames / (time.time() - global_start_time):.2f}"
        )

        if ground_truth:
            print(
                f"IoU score increase through fusion: {np.mean(avg_iou_after) - np.mean(avg_iou_before)}"
            )
            print(
                f"Average TP: {np.mean(avg_tp):.2f}, FP: {np.mean(avg_fp):.2f}, FN: {np.mean(avg_fn):.2f}"
            )

        cv2.destroyAllWindows()
