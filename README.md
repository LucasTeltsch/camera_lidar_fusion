#  Multi-Sensor Object Detection with Camera and LiDAR  
### High-Level Fusion Using the NuScenes Dataset and Real Vehicle Data

This repository contains the full implementation of a bachelor thesis focused on multi-sensor object detection using camera and LiDAR data. The project is split into two main components:

1. **Evaluation on the [NuScenes Dataset](https://www.nuscenes.org/nuscenes)**  
2. **Real-Time Sensor Fusion on a Research Vehicle**

---

## 🔧 Components

### 1. Evaluation on the NuScenes Dataset

- Runs as a standalone Python project (no ROS)
- Includes:
  - NuScenes data loader
  - YOLOv11 and 3D LiDAR-based object detectors
  - High-level fusion of 2D/3D bounding boxes
  - Visualization of detections and evaluation metrics (IoU, confidence, etc.)

📂 See: [`nuScenes_fusion/README.md`](nuScenes_fusion/README.md)

🖥 Run:
```bash
python main.py
``` 

### 2. Real-Time Fusion on Research Vehicle
- ROS 2-based implementation using real sensor data
- Uses: 
  - DALSA Genie Nano C4030 camera
  - Ouster OS1-128 LiDAR sensor
- Includes:
  - ROS nodes for camera/LiDAR input
  - ROS nodes for synchronisation of camera and LiDAR data
  - Detection pipelines and fusion logic
  - Launch infrastructure for deployment

📂 See: [`ros_fusion/README.md`](ros_fusion/README.md)