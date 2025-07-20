from queue import Queue

from nuscenes.nuscenes import NuScenes


class NuScenesDataLoader:
    """
    Class to handle loading data from the NuScenes dataset.
    This class initializes the NuScenes dataset and provides a method to load new data
    into a queue. The queue contains tuples of image and point cloud paths.
    """

    def __init__(self, nusc: NuScenes):
        """
        Initializes the NuScenes dataset.
        Counter for the number of samples loaded is set to zero,
        and the maximum number of samples is determined from the dataset.

        Args:
            nusc (NuScenes): An instance of the NuScenes dataset.
        """
        self.nusc = nusc

        self.count = 0
        # self.max_samples = len(self.nusc.sample)
        self.max_samples = 15

    def load_new_data(self, sync_data_queue: Queue):
        """
        Continuously loads new data from the NuScenes dataset and puts it into the queue.
        Image, lidar paths and the sample token are put into `sync_data_queue`. 
        This method will run indefinitely until the maximum number of samples is reached.
        Blocks if queues reaches their maximum size.

        Args:
            sync_data_queue (Queue): FIFO queue for storing image, point cloud paths and sample tokens
        """
        while True:
            if self.count >= self.max_samples:
                sync_data_queue.put((None, None, None))
                print("[Dataloader] Stopping. No more samples available.")
                break

            sample = self.nusc.sample[self.count]
            self.count += 1

            cam_token = sample["data"]["CAM_FRONT"]
            lidar_token = sample["data"]["LIDAR_TOP"]

            cam_data = self.nusc.get("sample_data", cam_token)
            img_path = self.nusc.dataroot + "/" + cam_data["filename"]

            lidar_data = self.nusc.get("sample_data", lidar_token)
            lidar_path = self.nusc.dataroot + "/" + lidar_data["filename"]

            sync_data_queue.put((img_path, lidar_path, sample))
