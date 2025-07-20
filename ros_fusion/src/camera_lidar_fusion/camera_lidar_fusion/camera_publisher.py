import rclpy
from rclpy.node import Node

from interfaces.msg import CameraData

import threading
import time
import cv2

from camera_lidar_fusion.camera import Camera


class CameraPublisher(Node):

    def __init__(self, fps):
        super().__init__('camera_publisher')
        self.get_logger().info('Camera Publisher has started')

        self.publisher = self.create_publisher(
            CameraData,
            'camera_data',
            10
        )

        self.camera = Camera()

        # Resize image for higher data frequency
        width = int(4112/4)
        height = int(2176/4)
        self.image_shape = (width, height)

        # Setup image buffer
        self.latest_image = None

        # Setup thread for continous image acquition
        self.image_lock = threading.Lock()
        self.acqusition_thread = threading.Thread(target=self.acquire_images, daemon=True)
        self.acqusition_thread.start()

        # Continously publish data
        self.create_timer(1.0 / fps, self.publish_data)

        self.time_past = time.time()
        self.frame_count = 0
        self.elapsed_time = 0

    def acquire_images(self):
        """ Continously acquire images in a separate thread """
        while rclpy.ok():
            image, _ = self.camera.get_next_frame()
            with self.image_lock:
                self.latest_image = image
            time.sleep(0.001)  # Add short sleep to avoid maxing out CPU

    def publish_data(self):
        """ Pulbish the most recent image at the specific frame rate"""
        current_time = time.time()
        frame_duration = current_time - self.time_past
        self.elapsed_time += frame_duration
        camera_data = CameraData()

        # Retrieve latest image and publish
        with self.image_lock:
            if self.latest_image is not None:
                self.frame_count += 1
                img = self.latest_image
                img = cv2.resize(img, self.image_shape)

                # Serialize image
                camera_data.camera_data = img.ravel().tolist() 
                self.publisher.publish(camera_data)
                self.get_logger().info('Publishing image')

        if self.frame_count == 10:
            avg_fps = self.frame_count / self.elapsed_time
            self.get_logger().info(
                f'Average FPS over last 10 frames: {avg_fps:.2f}')

            # Reset counters
            self.frame_count = 0
            self.elapsed_time = 0

        # Update time for FPS calculation
        self.time_past = current_time


def main(args=None):
    rclpy.init(args=args)
    fps = 1
    camera_publisher = CameraPublisher(fps)

    try:
        rclpy.spin(camera_publisher)
    finally:
        camera_publisher.destroy()
        rclpy.shutdown()
