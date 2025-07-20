import argparse
import os
import sys

import rclpy.utilities
import yaml
from dotmap import DotMap


class Camera:
    def __init__(self):
        self._dalsa_cam = self.initCam()

    def get_next_frame(self):
        img, status = self._dalsa_cam.get_next_frame()
        return img, status

    def initCam(self):
        filtered_args = rclpy.utilities.remove_ros_args(sys.argv)

        parser = argparse.ArgumentParser(
            prog="Dalsa Camera Display", description="Shows dalsa camera live stream."
        )
        parser.add_argument(
            "-c", "--config", default="config.yaml", help="load config file"
        )
        args = parser.parse_args(filtered_args[1:])
        config = DotMap(self.loadConfig(args.config))

        from camera_lidar_fusion.gigev_common.gigev import GigEV

        cam = GigEV()
        cam.camera_list()
        cam.open_camera(config.camera.id)
        return cam

    def loadConfig(self, filepath):
        if os.path.isfile(filepath):
            with open(filepath, "r") as file:
                config = yaml.safe_load(file)
        else:
            config = self.defaultConfig()
            with open(filepath, "w") as file:
                yaml.dump(config, file)
        return config

    def defaultConfig(self):
        return {"camera": {"id": "H2352648", "img_width": 4112, "img_height": 2176}}


if __name__ == "__main__":
    cam = Camera()
