#!/usr/bin/python3
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='camera_lidar_fusion',
            executable='server',
            name='server'
        ),
        Node(
            package='camera_lidar_fusion',
            executable='client',
            name='client'
        ),
        Node(
            package='camera_lidar_fusion',
            executable='camera_publisher',
            name='camera_publisher'
        ),
        Node(
            package='camera_lidar_fusion',
            executable='lidar_publisher',
            name='lidar_publisher'
        )
        
    ])
