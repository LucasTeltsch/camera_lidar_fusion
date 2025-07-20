import time
from queue import Queue

from detectors.img_obj_detector import ImageObjectDetector
from detectors.pc_obj_detector import PointCloudObjectDetector


class ObjectDetector:
    """
    A class for detecting objects in images and point clouds using YOLO and mmdetection3d models.
    This class initializes the object detectors for both image and point cloud data,
    and provides a method to detect objects in synchronized data from both sources.
    """

    def __init__(self, img_model: str, pc_model: str, verbose=False):
        """
        Initializes the ObjectDetector with specified YOLO and mmdetection3d models.
        Loads the YOLO model for image object detection and a mmdetection3d model for point cloud object detection.
        This method sets up the necessary object detectors for processing images and point clouds.

        Args:
            img_model (str): Description of the image model to be used for object detection.
            pc_model (str): Description of the point cloud model to be used for object detection.
            verbose (bool): Displays additional console logs
        """

        self.img_model = img_model
        self.pc_model = pc_model

        self.img_obj_detector = ImageObjectDetector(img_model, verbose)
        self.pc_obj_detector = PointCloudObjectDetector(pc_model, verbose)

    def detect_objects(self, sync_data_queue: Queue, post_process_queue: Queue, presenation_mode=False):
        """
        Detects objects in synchronized data from both image and point cloud sources.
        This method continuously retrieves data from the `sync_data_queue`, processes the images and point clouds
        using the respective object detectors, and puts the results into the `post_process_queue`.
        For visualization purposes, the image and point cloud paths and sample are also passed on in the queue.
        This method runs indefinitely until it receives a termination signal (None, None, None) in the queue.

        Args:
            sync_data_queue (Queue): Queue containing synchronized data for image and point cloud paths
            post_process_queue (Queue): Queue to store the detection results from both image and point cloud object detectors.
            presentation_mode (bool): Stops the visualization at the first image and does not display fps metrics
        """

        start_time = 0

        while True:
            data = sync_data_queue.get()

            if data == (None, None, None):
                sync_data_queue.task_done()
                post_process_queue.put((None, None, None, None, None))
                print("[ObjectDetector] Stopping. No more data available.")
                break

            img_path, pc_path, sample = data

            if not presenation_mode:
                start_time = time.time()
            
            img_results = self.img_obj_detector.detect_objects(img_path)
            pc_results = self.pc_obj_detector.detect_objects(pc_path)
            
            if not presenation_mode:
                print(f"[ObjectDetector] FPS for camera and lidar object detection: {1 / (time.time() - start_time):.2f}")

            sync_data_queue.task_done()

            post_process_queue.put((img_path, pc_path, img_results, pc_results, sample))
            
        print(f"[ObjectDetector] Average FPS for object detection: {1 / (time.time() - start_time):.2f}")
