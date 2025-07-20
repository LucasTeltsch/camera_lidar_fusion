from mmdet3d.apis import LidarDet3DInferencer


class PointCloudObjectDetector:
    """
    A class for detecting objects in point clouds using a configured object detection model from mmdetection3d.
    """

    def __init__(self, config: dict, verbose=False):
        """
        Initializes the PointCloudObjectDetector with detection model configuration.
        Loads the model configuration and checkpoint files
        and sets up the inferencer for object detection.

        Args:
            config (dict): A dictionary containing the configuration and checkpoint files for the object detection model.
            verbose (bool): Displays additional console logs
        """
        if "config_file" not in config or "checkpoint_file" not in config:
            raise ValueError(
                "centerpoint_config must contain 'config_file' and 'checkpoint_file'"
            )

        config_file = config["config_file"]
        checkpoint_file = config["checkpoint_file"]

        self.inferencer = LidarDet3DInferencer(config_file, checkpoint_file)
        self.inferencer.show_progress = False

        self.verbose = verbose

        if self.verbose:
            print("[PointCloudObjectDetector] Successfully loaded CenterPoint model")

    def detect_objects(self, point_cloud_path):
        """
        Detects objects in a point cloud using the configured model.
        This method processes the point cloud file specified by `point_cloud_path`
        and returns the detection results.

        Args:
            point_cloud_path (str): Path to the point cloud file to be processed.

        Returns:
            _type_: Detection results from the configured model.
        """

        if self.verbose:
            print(f"Processing point cloud: {point_cloud_path}")

        inputs = dict(points=point_cloud_path)

        results = self.inferencer(inputs, return_datasamples=False)

        results = results["predictions"][0]

        return results
