# Camera and LiDAR Object detection with High-Level Fusion on the NuScenes Dataset

This repository contains code developed as part of a bachelor thesis on multi-sensor object detection. It implements camera and LiDAR-based object detection pipelines and combines their outputs using high-level fusion techniques. The project uses the [NuScenes dataset](https://www.nuscenes.org/nuscenes) as the primary source for evaluation.

## Features
- **Camera Object Detection**  
  Uses a deep learning-based detector (YOLOv11) on RGB images to detect vehicles, pedestrians, and other road users.

- **LiDAR Object Detection**  
  Processes point cloud data with a 3D detection network (based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)) to localize objects in 3D space.

- **High-Level Fusion**  
  Fuses the independent camera and LiDAR detections based on geometric proximity and detection confidence to improve robustness.

## Repository Structure

```
.
├── main.py                         # Entry point of the application
├── data_fuser.py                   # Performs sensor fusion and result visualization
├── nuscenes_data_loader.py         # Loads and processes nuScenes dataset
├── INSTALL.md                      # Installation instructions
├── README.md                       # Project overview and usage
│   
├── detectors/                      # Object detection modules
│   ├── obj_detector.py             # Initializes image and point cloud detectors
│   ├── img_obj_detector.py         # Image-based object detector (YOLO)
│   └── pc_obj_detector.py          # Point cloud-based object detector (e.g., CenterPoint, PointPillars)
│   
├── lidar_config/                   # Configs and pre-trained models for lidar-based detectors
│   ├── checkpoints/                # Pre-trained weights
│   │   ├── centerpoint/    
│   │   ├── pointpillars/   
│   │   └── ssn/    
│   └── configs/                    # Model configurations
│       ├── _base_/
│       ├── centerpoint/
│       ├── pointpillars/
│       └── ssn/
│
├── utils/                          # Utility functions
│   ├── draw_utils.py               # Drawing 2D/3D bounding boxes on images
│   ├── fusion_utils.py             # Fusion of 2D and 3D detections
│   ├── object_description_util.py  # Label mapping and coloring for nuScenes and COCO
│   └── projection_utils.py         # Projects 3D boxes onto the image plane
│
└── yolo_config/                    # YOLOv11 configuration
    └── yolo11n.pt                  # Pre-trained YOLOv11 model weights

```

## Usage
```
python main.py
```

```
optional arguments:
  -h, --help            show this help message and exit
  --compute_average     Computes average 3D bounding boxes for matched objects
  --show_depth          Visualizes depth information on image plane
  --presentation_mode   Stops the visualization at the first image and does not display fps metrics
  --verbose             Displays more information
  --camera_only         Only displays 2D bounding boxes from the camera object detection
  --lidar_only          Only displays 3D bounding boxes from the lidar object detection
  --average_only        Only displays average 3D bounding boxes from matched objects
  --show_iou_score      Displays the IoU score for matched objects
  --display_highest_certainty
                        On matches, display either 2D or 3D bounding box based on precision score
  --ground_truth        Compares the 3D bounding boxes to the ground truth
  --draw_gt_boxes       Draws ground truth boxes on the image
```

## Installation
Please refer to [Installation](INSTALL.md) for installation instructions.

