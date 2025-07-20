import ctypes
import sys
from PIL import Image
import numpy
import camera_lidar_fusion.gigev_common.pygigev as pygigev


def ip_addr_to_string(ip):
    "Convert 32-bit integer to dotted IPv4 address."
    return ".".join(map(lambda n: str(ip>>n & 0xFF), [24,16,8,0]))


class GigEV:
    def __init__(self):
        pygigev.GevApiInitialize()
        self.cam_info = None
        self.cam_id_list = dict()
        self.cam_id = None
        self.cam_handle = None
        self.payload_size = None
        self.image_size = None
        self.buffer_addr = ((ctypes.c_void_p)*1)()
        self.gigev_buffer_ptr = ctypes.POINTER(pygigev.GEV_BUFFER_OBJECT)()
        self.gigev_timeout = (ctypes.c_uint32)(1000)

    def camera_list(self, max_cameras=16):
        # Allocate a maximum number of camera info structures.
        num_cam_found = (ctypes.c_uint32)(0)
        cam_info = (pygigev.GEV_CAMERA_INFO * max_cameras)()
        
        # Get the camera list
        status = pygigev.GevGetCameraList(cam_info, max_cameras, ctypes.byref(num_cam_found) )
        self.cam_info = cam_info
        
        if ( status != 0  ):
            print("\nError ", status,"getting camera list - exitting")
            quit()
        
        # Proceed
        print("\nCamera list:")
        if (num_cam_found.value == 0):
            print("No cameras found - exitting")
            quit()
        
        for cam_index in range(num_cam_found.value):
            name = f"{self.cam_info[cam_index].manufacturer.decode('UTF-8')} " \
                   f"{self.cam_info[cam_index].model.decode('UTF-8')} | No: " \
                   f"{self.cam_info[cam_index].serial.decode('UTF-8')}"
            ip = f"Camera IP: {ip_addr_to_string(self.cam_info[cam_index].ipAddr)} | " \
                 f"NIC IP: {ip_addr_to_string(self.cam_info[cam_index].host.ipAddr)}"
            # Prints might not work with ROS
            # print(f"\t{name}\t{ip}")
            self.cam_id_list[self.cam_info[cam_index].serial.decode('UTF-8')] = cam_index
        
    def open_camera(self, cam_id):
        self.cam_id = cam_id
        cam_index = self.cam_id_list[self.cam_id]
        self.cam_handle = (ctypes.c_void_p)()
        status = pygigev.GevOpenCamera(self.cam_info[cam_index], pygigev.GevExclusiveMode, ctypes.byref(self.cam_handle))
        print(f"\nCamera {self.cam_id} Ready (Status: {status})")

        self._setup_buffer()
        self._start_transfer()

    def close_camera(self):
        self._stop_transfer()
        status = pygigev.GevCloseCamera(ctypes.byref(self.cam_handle))
        print(f"\nCamera {self.cam_id} Closed (Status: {status})")
        
    def _setup_buffer(self):
        # Get the payload parameters
        print("Image buffer setup")
        self.payload_size = (ctypes.c_uint64)()
        pixel_format = (ctypes.c_uint32)()
        
        pygigev.GevGetPayloadParameters(self.cam_handle, ctypes.byref(self.payload_size), ctypes.byref(pixel_format))
        pixel_format_unpacked = pygigev.GevGetUnpackedPixelType(pixel_format)
        
        print(f"\tPayload Size: {self.payload_size.value}\n" \
              f"\tPixel Format: {hex(pixel_format.value)} | Pixel Format Unpacked: {hex(pixel_format_unpacked)}")
        
        # Get the Width and Height (extra information)
        feature_strlen = (ctypes.c_int)(pygigev.MAX_GEVSTRING_LENGTH)
        unused = (ctypes.c_int)(0)
        width_str = ((ctypes.c_char)*feature_strlen.value)()
        height_str = ((ctypes.c_char)*feature_strlen.value)()
        
        pygigev.GevGetFeatureValueAsString(self.cam_handle, b'Width', unused, feature_strlen, width_str)
        pygigev.GevGetFeatureValueAsString(self.cam_handle, b'Height', ctypes.byref(unused), feature_strlen, height_str)
        
        self.image_size = (int(width_str.value), int(height_str.value))
        print(f"\tImage Size: {width_str.value.decode('UTF-8')}x{height_str.value.decode('UTF-8')}px")
        
        # Allocate buffers to store images.
        # (Handle cases where image is larger than payload due to pixel unpacking)
        buffer_size = self.payload_size.value
        pixel_size_byte = pygigev.GevGetPixelSizeInBytes(pixel_format_unpacked)
        buffer_size_unpacked = int(width_str.value) * int(height_str.value) * pixel_size_byte
        
        if (buffer_size_unpacked > buffer_size):
            buffer_size = buffer_size_unpacked
        
        self.buffer_addr[0] = ctypes.cast(((ctypes.c_char)*buffer_size)(), ctypes.c_void_p)
        print(f"\tBuffer Addr: {hex(self.buffer_addr[0])} | Buffer Size: {buffer_size}")
        
    def _start_transfer(self):
        pygigev.GevInitializeTransfer(self.cam_handle, pygigev.Asynchronous, self.payload_size, 1, self.buffer_addr)
        pygigev.GevStartTransfer(self.cam_handle, -1)
        
    def _stop_transfer(self):
        pygigev.GevFreeTransfer(self.cam_handle)
        
    def get_next_frame(self):
        while True:
            status = pygigev.GevWaitForNextFrame(self.cam_handle, ctypes.byref(self.gigev_buffer_ptr), self.gigev_timeout.value)
            if self.gigev_buffer_ptr:
                break
        gigev_buffer = self.gigev_buffer_ptr.contents
        
        if status == 0:
            if gigev_buffer.status == 0 :
                #print(f"Image: [id: {gigev_buffer.id}, w: {gigev_buffer.w}, h: {gigev_buffer.h}, " \
                #      f"address: {hex(gigev_buffer.address)}, status: {gigev_buffer.status}")
                img_addr = ctypes.cast(gigev_buffer.address, ctypes.POINTER(ctypes.c_ubyte * gigev_buffer.recv_size))
                pil_img = Image.frombuffer('L', self.image_size, img_addr.contents, 'raw', 'L', 0, 1)
                image = numpy.array(pil_img)
            else:
                print(f"GigEV Buffer has Status: {gigev_buffer.status}")
        else:
            print(f"Error Camera Status: {status}")
        
        return image, status

    def __del__(self):
        pygigev.GevApiUninitialize()
