### Camera and LiDAR Object detection with High-Level Fusion with Sensor Data from the Research Vehicle


# Camera and LiDAR Object detection with High-Level Fusion on the NuScenes Dataset

This repository contains code developed as part of a bachelor thesis on multi-sensor object detection. It implements camera and LiDAR-based object detection pipelines and combines their outputs using high-level fusion techniques. The project uses acutal sensor data from a research vehicle. The sensors are a DALSA Genie Nano C4030 camera and a Ouster OS1-128 LiDAR sensor. 

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
├── INSTALL.md                                  # Installation instructions
├── README.md                                   # Project overview and usage
└── src/                                        # ROS 2 workspace source directory
    ├── camera_lidar_fusion/                    # Main package: nodes, fusion logic, detection models
    │   ├── camera_lidar_fusion/
    │   │   ├── camera.py                       # Camera initialization helper
    │   │   ├── camera_publisher.py             # Publishes image data
    │   │   ├── lidar_publisher.py              # Publishes LiDAR point clouds
    │   │   ├── server.py                       # Synchronizes and relays sensor data
    │   │   ├── client.py                       # Client for fused data
    │   │   ├── data_fuser.py                   # Fusion logic for synchronized detections
    │   │   ├── detectors/                      # Object detection modules
    │   │   │   ├── obj_detector.py             # Initializes image & point cloud detectors
    │   │   │   ├── img_obj_detector.py         # Image-based detector (e.g., YOLO)
    │   │   │   └── pc_obj_detector.py          # Point cloud-based detector (e.g., CenterPoint)
    │   │   ├── lidar_config/                   # LiDAR configs and pre-trained models
    │   │   │   ├── checkpoints/                # Pre-trained model weights
    │   │   │   └── configs/                    # Model configuration files
    │   │   ├── yolo_config/                    # YOLOv11 weights
    │   │   │   └── yolo11n.pt
    │   │   ├── utils/                          # Utility functions
    │   │   │   ├── draw_utils.py               # Draws bounding boxes
    │   │   │   ├── fusion_utils.py             # Fuses 2D and 3D detections
    │   │   │   ├── object_description_util.py  # Label and color handling
    │   │   │   └── projection_utils.py         # Projects 3D boxes to 2D image
    │   └── ...                                 # Standard ROS files (setup.py, package.xml, etc.)
    │
    ├── interfaces/                             # Custom ROS message definitions
    │   ├── msg/            
    │   │   ├── CameraData.msg                  # Image message
    │   │   ├── LidarData.msg                   # Point cloud message
    │   │   └── SyncData.msg                    # Fused message format
    │   └── ...
    │
    └── launch_fusion/                          # ROS 2 launch files
        └── launch.py                           # Entry point for launching all nodes
```

## Usage
```
source /opt/ros/galactic/setup.bash
colcon build
source install/setup.bash

ros2 launch launch_fusion launch.py
```

# Installation
Please refer to [Installation](INSTALL.md) for installation instructions.