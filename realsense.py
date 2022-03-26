import pyrealsense2 as rs
import numpy as np


class RSCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.width = 640
        self.height = 480
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width,
                             self.height, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, self.width,
                             self.height, rs.format.z16, 30)

        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        image_sensor = profile.get_device().query_sensors()[1]

        self.depth_scale = depth_sensor.get_depth_scale()
        image_sensor.set_option(rs.option.enable_auto_exposure, True)
        print("depth scale is" + str(self.depth_scale))

    def get_raw_frames(self):
        frames = self.pipeline.wait_for_frames()

        depth = np.asanyarray(frames.get_depth_frame().get_data())
        color = np.asanyarray(frames.get_color_frame().get_data())

        return (color, depth)

    def get_rs_frames(self):
        frames = self.pipeline.wait_for_frames()

        depth = frames.get_depth_frame()
        color = frames.get_color_frame()

    def get_color_aligned_frames(self):
        frames = self.pipeline.wait_for_frames()

        align_to = rs.stream.color
        align = rs.align(align_to)
        aligned_frames = align.process(frames)

        depth = np.asanyarray(aligned_frames.get_depth_frame().get_data())
        color = np.asanyarray(aligned_frames.get_color_frame().get_data())

        return (color, depth)

    def release(self):
        self.pipeline.stop()
