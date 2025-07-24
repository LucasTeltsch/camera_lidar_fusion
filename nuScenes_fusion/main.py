import argparse
from queue import Queue
from threading import Thread

from data_fuser import DataFuser
from detectors.obj_detector import ObjectDetector
from nuscenes.nuscenes import NuScenes
from nuscenes_data_loader import NuScenesDataLoader


def get_detector_config():
    # Set configurations for YOLO and CenterPoint
    yolo_config = "yolo_config/yolo11n.pt"  # Path to your YOLOv11 model
    mmlab_config = {
        "config_file": "lidar_config/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py",
        "checkpoint_file": "lidar_config/checkpoints/pointpillars/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626-c3f0483e.pth",
    }
    # mmlab_config = {
    #     "config_file": "lidar_config/configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py",
    #     "checkpoint_file": "lidar_config/checkpoints/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth",
    # }  
    return yolo_config, mmlab_config


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compute_average",
        action="store_true",
        help="Computes average 3D bounding boxes for matched objects",
    )
    parser.add_argument(
        "--show_depth",
        action="store_true",
        help="Visualizes depth information on image plane",
    )
    parser.add_argument(
        "--presentation_mode",
        action="store_true",
        help="Stops the visualization at the first image and does not display fps metrics",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Displays additional information"
    )
    parser.add_argument(
        "--camera_only",
        action="store_true",
        help="Only displays 2D bounding boxes from the camera object detection",
    )
    parser.add_argument(
        "--lidar_only",
        action="store_true",
        help="Only displays 3D bounding boxes from the lidar object detection",
    )
    parser.add_argument(
        "--average_only",
        action="store_true",
        help="Only displays average 3D bounding boxes from matched objects",
    )
    parser.add_argument(
        "--show_iou_score",
        action="store_true",
        help="Displays the IoU score for matched objects",
    )
    parser.add_argument(
        "--display_highest_certainty",
        action="store_true",
        help="On matches, display either 2D or 3D bounding box based on precision score",
    )
    parser.add_argument(
        "--ground_truth",
        action="store_true",
        help="Compares the 3D bounding boxes to the ground truth",
    )
    parser.add_argument(
        "--draw_gt_boxes",
        action="store_true",
        help="Draws ground truth boxes on the image",
    )

    args = parser.parse_args()
    return args


sync_data_queue = Queue(maxsize=10)

post_process_queue = Queue(maxsize=10)


def main():

    args = parse_arguments()

    nusc = NuScenes(
        version="v1.0-mini",
        dataroot="/home/lucas/NuScenes/data",
        verbose=args.verbose,
    )

    yolo_config, mmlab_config = get_detector_config()

    nuScenes_data_loader = NuScenesDataLoader(nusc)

    obj_detector = ObjectDetector(yolo_config, mmlab_config, args.verbose)

    data_fuser = DataFuser(nusc)

    data_thread = Thread(
        target=nuScenes_data_loader.load_new_data, args=(sync_data_queue,), daemon=True
    )

    obj_detector_thread = Thread(
        target=obj_detector.detect_objects,
        args=(sync_data_queue, post_process_queue, args.presentation_mode),
        daemon=True,
    )

    data_thread.start()
    obj_detector_thread.start()

    # Run GUI-related in main thread
    data_fuser.fuse_data_and_display(
        nusc,
        post_process_queue,
        args.compute_average,
        args.show_depth,
        args.presentation_mode,
        args.camera_only,
        args.lidar_only,
        args.average_only,
        args.show_iou_score,
        args.display_highest_certainty,
        args.ground_truth,
        args.draw_gt_boxes,
    )

    data_thread.join()
    obj_detector_thread.join()

    print("[Main] All threads have completed execution.")


if __name__ == "__main__":
    main()
