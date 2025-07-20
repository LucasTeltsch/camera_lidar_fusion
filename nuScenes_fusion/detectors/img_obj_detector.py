import cv2
import torch
from ultralytics import YOLO


class ImageObjectDetector:
    def __init__(self, model_path: str, verbose=False):
        """
        Initializes the ImageObjectDetector with a YOLO model.
        Loads the YOLO model from the specified path and sets it to use GPU if available.

        Args:
            model_path (str): Path to the YOLO model file.
            verbose (bool): Displays additional console logs
        """
        if not model_path:
            raise ValueError("model_path must be provided")

        self.model = YOLO(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        self.verbose = verbose

        if self.verbose: 
            print("[ImageObjectDetector] Successfully loaded YOLO model")

    def detect_objects(self, img_path: str):
        """
        Detects objects in an image using the YOLO model.
        This method processes the image file specified by `img_path`
        and returns the detection results.

        Args:
            img_path (str): Path to the image file to be processed.

        Returns:
            _type_: Detection results from the YOLO model.
        """

        if self.verbose: 
            print(f"Processing image: {img_path}")

        image = cv2.imread(img_path)

        results = self.model(image, verbose=False)[0].boxes.data

        return results
