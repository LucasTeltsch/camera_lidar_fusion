from queue import Queue
from threading import Thread

import numpy as np
import rclpy
from interfaces.msg import SyncData
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from camera_lidar_fusion.data_fuser import DataFuser
from camera_lidar_fusion.detectors.obj_detector import ObjectDetector


class FusionClient(Node):
    def __init__(self):
        super().__init__("fusion_client")
        self.get_logger().info("FusionClient has started")

        self.image_width = int(4112 / 4)
        self.image_height = int(2176 / 4)
        # self.original_image_shape = (self.image_width, self.image_height)
        self.original_image_shape = (900, 1600, 3)  # For test image

        self.lidar_width = 1024
        self.lidar_height = 128
        self.original_pc_shape = (self.lidar_width, self.lidar_height)
        self.original_pc_shape = (131072, 5)

        self.sync_data_subscription = self.create_subscription(
            SyncData, "sync_data", self.sync_data_callback, 10
        )

        self.sync_data_queue = Queue(maxsize=10)
        self.post_process_queue = Queue(maxsize=10)

        yolo_config, mmlab_config = self.get_detector_config()

        # Threads for Object Detection, Fusion and Visualiziation
        obj_detector = ObjectDetector(yolo_config, mmlab_config)

        data_fuser = DataFuser()

        obj_detector_thread = Thread(
            target=obj_detector.detect_objects,
            args=(
                self.get_logger(),
                self.sync_data_queue,
                self.post_process_queue,
            ),
            daemon=True,
        )

        data_fuser_thread = Thread(
            target=data_fuser.fuse_data_and_display,
            args=(self.post_process_queue,),
            daemon=True,
        )

        obj_detector_thread.start()
        data_fuser_thread.start()

        self.get_logger().info("FusionClient initialized successfully")

    def sync_data_callback(self, sync_data: SyncData):
        self.get_logger().info("Image and Point Cloud received")
        camera_data = sync_data.camera_data
        lidar_data = sync_data.lidar_data

        # Reshape data into their original form
        image = self.reshape_data(
            self.get_logger(),
            camera_data,
            self.original_image_shape,
            dtype=np.uint8,
        )

        point_cloud = self.reshape_data(
            self.get_logger(), lidar_data, self.original_pc_shape, dtype=np.uint32
        )

        self.sync_data_queue.put((image, point_cloud))

        self.get_logger().info(f"Sync_data_queue Size: {self.sync_data_queue.qsize()}")

    def get_detector_config(self):
        # Set configurations for YOLO and CenterPoint
        yolo_config = "src/camera_lidar_fusion/camera_lidar_fusion/yolo_config/yolo11n.pt"  # Path to your YOLOv11 model
        mmlab_config = {
            "config_file": "src/camera_lidar_fusion/camera_lidar_fusion/lidar_config/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py",  
            "checkpoint_file": "src/camera_lidar_fusion/camera_lidar_fusion/lidar_config/checkpoints/pointpillars/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626-c3f0483e.pth", 
        }
        return yolo_config, mmlab_config

    def reshape_data(self, logger, data, original_shape, dtype):
        data = np.array(data, dtype)
        try:
            data = data.reshape(original_shape)
        except Exception as e:
            logger.info(f"{e}")

        return data


def main(args=None):
    rclpy.init(args=args)
    fusion_client = FusionClient()
    executor = MultiThreadedExecutor()
    executor.add_node(fusion_client)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        fusion_client.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
