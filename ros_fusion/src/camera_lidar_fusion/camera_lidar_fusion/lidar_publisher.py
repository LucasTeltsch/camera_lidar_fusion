import time
from threading import Lock, Thread

import numpy as np
import rclpy
from interfaces.msg import LidarData
from ouster.sdk import client
from ouster.sdk.client import LidarScan
from ouster.sdk.client.data import XYZLut
from rclpy.node import Node


class LidarPublisher(Node):
    def __init__(self, frames_per_second):
        super().__init__("lidar_publisher")
        self.get_logger().info("Lidar Publisher has started")

        # ROS publisher for LidarData messages
        self.publisher = self.create_publisher(LidarData, "lidar_data", 10)

        # Initialize sensor parameters
        self.hostname = "10.131.67.114"
        self.lidar_port = 7502
        self.imu_port = 7503

        # Initialize sensor and scans
        self.sensor = client.Sensor(self.hostname, self.lidar_port, self.imu_port)
        self.metadata = self.sensor.metadata
        self.scans = iter(client.Scans(self.sensor))
        self.xyzlut = XYZLut(info=self.metadata)

        # Variables to store the latest scan and synchronization lock
        self.latest_scan = None
        self.scan_lock = Lock()

        # Start the acquisition thread
        self.acquisition_thread = Thread(target=self.acquire_scans)
        self.acquisition_thread.daemon = True
        self.acquisition_thread.start()

        # Publishing timer for the desired frame rate
        self.timer = self.create_timer(1.0 / frames_per_second, self.publish_data)

        # For FPS calculation
        self.time_past = time.time()
        self.frame_count = 0
        self.elapsed_time = 0

    def acquire_scans(self):
        """Continuously acquire Lidar scans in a separate thread"""
        while rclpy.ok():

            try:
                scan = next(self.scans)
                range_data = scan.field(client.ChanField.RANGE)
                reflectivity = scan.field(client.ChanField.REFLECTIVITY)

                xyz_points = self.xyzlut(range_data).astype(np.float32)
                xyz_points = xyz_points.reshape(-1, 3)
                xyz_points = xyz_points[~np.isnan(xyz_points).any(axis=1)]
                reflectivity = reflectivity.astype(np.float32).reshape(-1, 1)
                reflectivity = reflectivity[~np.isnan(reflectivity).any(axis=1)]
                timestamp = np.zeros((xyz_points.shape[0], 1), dtype=np.float32)
                xyz_points = np.hstack(
                    [xyz_points, reflectivity, timestamp]
                )  # Shape [N, 5]

                with self.scan_lock:
                    self.latest_scan = xyz_points
            except StopIteration:
                self.get_logger().error("Failed to get next Lidar scan.")
                break
            time.sleep(0.001)  # Short sleep to avoid maxing out CPU

    def publish_data(self):
        """Publish the most recent Lidar scan at the specified frame rate"""
        current_time = time.time()
        frame_duration = current_time - self.time_past
        self.elapsed_time += frame_duration
        self.frame_count += 1

        # Publish the latest scan if available
        with self.scan_lock:
            if self.latest_scan is not None:
                scan = self.serialize_scan(self.latest_scan)
                self.publisher.publish(scan)
                self.get_logger().info("Publishing lidar data...")

        # Calculate average FPS every 10 frames
        if self.frame_count == 10:
            avg_fps = self.frame_count / self.elapsed_time
            self.get_logger().info(
                f"LIDAR: Average FPS over last 10 frames: {avg_fps:.2f}"
            )

            # Reset counters for next interval
            self.frame_count = 0
            self.elapsed_time = 0

        # Update time for FPS calculation
        self.time_past = current_time

    def serialize_scan(self, scan: LidarScan):
        """Serialize the scan data to LidarData message format"""
        range_data = scan.field("RANGE").astype(np.uint32).ravel().tolist()
        
        lidar_data = LidarData()
        lidar_data.range_data = range_data

        return lidar_data


def main(args=None):
    rclpy.init(args=args)
    frames_per_second = 1  # Set your desired publish rate
    lidar_publisher = LidarPublisher(frames_per_second)

    try:
        rclpy.spin(lidar_publisher)
    finally:
        lidar_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
