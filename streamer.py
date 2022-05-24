import socket
import time
import cv2
import numpy as np
from realsense import RSCamera
import imagezmq


# sender_color = imagezmq.ImageSender(connect_to='tcp://10.39.60.39:5555')
# sender_depth = imagezmq.ImageSender(connect_to='tcp://10.39.60.39:5555')
# cam = RSCamera()

# jpeg_quality = 95
# hostname = socket.gethostname()

# while True:
#     frame, depth = cam.get_rs_color_aligned_frames()
#     # depth_colormap = cam.colorize_frame(depth)

#     frame = np.asarray(frame.get_data())
#     depth_frame = np.asarray(depth.get_data())

#     ret, jpg_frame = cv2.imencode(
#         '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
#     ret, jpg_frame_depth = cv2.imencode(
#         '.jpg', depth_frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

#     sender_color.send_jpg(hostname, jpg_frame)
#     sender_depth.send_jpg(hostname + '_depth', jpg_frame_depth)


class Streamer:
    def __init__(self) -> None:
        self.sender_color = imagezmq.ImageSender(
            connect_to='tcp://10.10.10.232:5555')
        self.sender_depth = imagezmq.ImageSender(
            connect_to='tcp://10.10.10.232:5555')
        self.hostname = socket.gethostname()
        self.jpeg_quality = 95

    def send_frames(self, color, depth):
        ret, jpg_frame = cv2.imencode(
            '.jpg', color, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        ret, jpg_frame_depth = cv2.imencode(
            '.jpg', depth, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        self.sender_color.send_jpg(self.hostname, jpg_frame)
        self.sender_depth.send_jpg(self.hostname + '_depth', jpg_frame_depth)
        print('Sent frames')
