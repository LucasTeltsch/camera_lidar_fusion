from threading import Lock

import rclpy
from interfaces.msg import CameraData, LidarData, SyncData
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class SyncServer(Node):
    def __init__(self, fps):
        super().__init__("sync_server")
        self.get_logger().info("SyncServer has started")

        # Separate locks for each sensor
        self.camera_lock = Lock()
        self.lidar_lock = Lock()

        # Shared data for sensors
        self.camera_data = None
        self.lidar_data = None

        # Subscription setup
        self.camera_subscription = self.create_subscription(
            CameraData, "camera_data", self.camera_callback, 10
        )
        self.lidar_subscription = self.create_subscription(
            LidarData, "lidar_data", self.lidar_callback, 10
        )
        self.sync_data_publisher = self.create_publisher(SyncData, "sync_data", 10)

        self.create_timer(1.0 / fps, self.publish_sync_data)

    def camera_callback(self, msg: CameraData):
        with self.camera_lock:  # Lock specific to camera data
            self.get_logger().info(
                f"IMAGE Data received: length={len(msg.camera_data)}"
            )
            self.camera_data = msg  # Safely update camera data

    def lidar_callback(self, msg: LidarData):
        with self.lidar_lock:  # Lock specific to lidar data
            self.get_logger().info(f"LIDAR Data received: length={len(msg.range_data)}")
            self.lidar_data = msg  # Safely update lidar data

    def publish_sync_data(self):
        """Tries continuesly to retrieve the locks and publishes data if camera_data and lidar_data is set"""
        with self.camera_lock, self.lidar_lock:
            if not (self.camera_data and self.lidar_data):
                return  # Simply skip if data isn't ready yet

            # Copy data
            camera_data = self.camera_data.camera_data
            lidar_data = self.lidar_data.range_data

            # Clear for next cycle
            self.camera_data = None
            self.lidar_data = None

        # Publish outside the lock
        sync_data = SyncData()
        sync_data.camera_data = camera_data
        sync_data.lidar_data = lidar_data
        self.sync_data_publisher.publish(sync_data)
        self.get_logger().info("Publishing SyncData")


def main(args=None):
    rclpy.init(args=args)
    sync_server = SyncServer(10)
    # Assign multiple threads to node
    executor = MultiThreadedExecutor()
    executor.add_node(sync_server)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        sync_server.destroy_node()
    rclpy.shutdown()
