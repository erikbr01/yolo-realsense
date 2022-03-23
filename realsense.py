import pyrealsense2 as rs
import numpy as np


class RSCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.pipeline.start(config)
        sensor = self.pipeline.get_active_profile(
        ).get_device().query_sensors()[1]
        sensor.set_option(rs.option.enable_auto_exposure, True)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()

        depth = np.asanyarray(frames.get_depth_frame().get_data())
        color = np.asanyarray(frames.get_color_frame().get_data())

        return (color, depth)
