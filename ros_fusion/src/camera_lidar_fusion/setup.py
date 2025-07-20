from setuptools import find_packages, setup

package_name = "camera_lidar_fusion"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="lucas",
    maintainer_email="lucas.teltsch@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "server = camera_lidar_fusion.server:main",
            "client = camera_lidar_fusion.client:main",
            "camera_publisher = camera_lidar_fusion.camera_publisher:main",
            "lidar_publisher = camera_lidar_fusion.lidar_publisher:main",
        ],
    },
)
