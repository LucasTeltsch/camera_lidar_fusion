### Installation Guide

**This installation was tested on Ubuntu 20.04**

Install ROS galactic

conda create -n ros python=3.8
conda activate ros

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -U openmim
mim install mmengine
mim install 'mmcv==2.1.0'
mim install 'mmdet>=3.0.0'

pip install mmdet3d

pip install ultralytics


pip install ouster-sdk

pip install catkin_pkg