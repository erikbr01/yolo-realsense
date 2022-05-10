import cv2
from realsense import RSCamera
import numpy as np
from elements.yolo import OBJ_DETECTION
from logger import Logger
import time
from classes import coco
import math
import zmq
import detection_msg_pb2
from tracking_object import TrackingObject

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=20,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


class Detector:
    def __init__(self, weight_file) -> None:
        self.RECORD_COUNTER = self.get_record_counter('counter')
        self.LOG_NAME = 'testing'

        self.LOGFILE = f"logs/{self.LOG_NAME}_{self.RECORD_COUNTER}.csv"
        self.VIDEO_OUT_FILE = f'videos/{self.LOG_NAME}_{self.RECORD_COUNTER}.avi'
        self.VIDEO_OUT_DEPTH_FILE = f'videos/{self.LOG_NAME}_depth_{self.RECORD_COUNTER}.avi'
        self.VIDEO_OUT_RAW_FILE = f'videos/{self.LOG_NAME}_raw_{self.RECORD_COUNTER}.avi'
        self.WEIGHTS = weight_file

        self.ZMQ_SOCKET_ADDR = "tcp://localhost:5555"

        self.object_classes = coco

        self.object_colors = list(np.random.rand(80, 3)*255)

        self.logger = Logger()
        self.records = np.empty((0, self.logger.cols))

        # Initalize ZMQ connection
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.connect(self.ZMQ_SOCKET_ADDR)

    def truncate(self, number, digits) -> float:
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper

    def get_record_counter(self, file):
        # Determine the record counter
        with open(file, 'r+', encoding='utf8') as f:
            # Process lines and get first line, there should only be one line
            lines = (line.strip() for line in f if line)
            x = [int(float(line.replace('\x00', ''))) for line in lines]
            ret = x[0]

            # Delete all file contents
            f.truncate(0)

            # Write back to file beginning
            f.seek(0)
            f.write(str(ret + 1))

        return ret

    def detect_objects_cont(self, target_object):

        object_detector = OBJ_DETECTION(self.WEIGHTS, self.object_classes)

        cam = RSCamera()

        output = cv2.VideoWriter(self.VIDEO_OUT_FILE, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 10, (cam.width, cam.height))

        output_depth = cv2.VideoWriter(self.VIDEO_OUT_DEPTH_FILE, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 10, (cam.width, cam.height))

        output_raw = cv2.VideoWriter(self.VIDEO_OUT_RAW_FILE, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 10, (cam.width, cam.height))

        starting_time = time.time()
        frame_counter = 0
        elapsed_time = 0
        tracking_objects = []
        serial_msg = None

        try:
            while True:
                # To sync the frame capture with the motion capture data, we only capture frames when receiving something
                _ = self.socket.recv()

                frame, depth_frame = cam.get_rs_color_aligned_frames()
                depth_colormap = cam.colorize_frame(depth_frame)
                output_depth.write(depth_colormap)

                # We aligh depth to color, so we should use the color frame intrinsics
                cam_intrinsics = frame.profile.as_video_stream_profile().intrinsics

                # Now get raw frama data as tensors
                frame = np.asanyarray(
                    frame.get_data())
                depth_frame = np.asanyarray(
                    depth_frame.get_data())

                output_raw.write(frame)

                # Frame in grayscale for tracking
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                # Detection every x frames, otherwise tracking
                perform_detection = frame_counter % 10 == 0

                # perform_detection = True
                if perform_detection:
                    tracking_objects.clear()
                    print("YOLO DETECTION")
                    objs = object_detector.detect(frame)

                    for obj in objs:
                        # Get bounding box coordinates
                        (xmin, ymin), (xmax, ymax) = obj['bbox']
                        label = obj['label']
                        color = self.object_colors[self.object_classes.index(
                            label)]

                        # Create mask for bounding box area
                        bbox_mask = np.zeros(frame_gray.shape, dtype='uint8')
                        cv2.rectangle(bbox_mask, (xmin, ymin),
                                      (xmax, ymax), 255, -1)

                        # Get points to track on this object
                        tracking_points = cv2.goodFeaturesToTrack(
                            frame_gray, mask=bbox_mask, **feature_params)

                        # Try again if it returns none
                        if tracking_points is None:
                            tracking_points = cv2.goodFeaturesToTrack(
                                frame_gray, mask=bbox_mask, **feature_params)

                        tracking_objects.append(
                            TrackingObject(obj['bbox'], tracking_points))

                        # Visualisation
                        for pt in tracking_points:
                            x, y = pt.ravel()
                            frame = cv2.circle(
                                frame, (int(x), int(y)), 5, color, -1)

                else:
                    print("LK TRACKING")

                    for i, tr_obj in enumerate(tracking_objects):
                        old_points = tr_obj.points
                        new_points, status, err = cv2.calcOpticalFlowPyrLK(
                            old_frame_gray, frame_gray, old_points, None, **lk_params)

                        new_bbox, disc_points = tr_obj.update_bbox(
                            new_points, status, depth_frame, cam)
                        objs[i]['bbox'] = new_bbox

                        # Visualisation
                        label = objs[i]['label']
                        color = self.object_colors[self.object_classes.index(
                            label)]
                        for pt in new_points:
                            x, y = pt.ravel()
                            frame = cv2.circle(
                                frame, (int(x), int(y)), 5, color, -1)

                        for pt in disc_points:
                            frame = cv2.rectangle(
                                frame, (pt[0] - 2, pt[1] - 2), (pt[0] + 2, pt[1] + 2), (0, 0, 255), 2)

                # localizing in 3D and plotting
                for obj in objs:
                    print(obj)
                    label = obj['label']
                    score = obj['score']
                    [(xmin, ymin), (xmax, ymax)] = obj['bbox']
                    color = self.object_colors[self.object_classes.index(
                        label)]

                    center_x = (xmax - xmin)/2 + xmin
                    center_y = (ymax - ymin)/2 + ymin

                    if center_y < cam.height and center_x < cam.width:
                        depth = depth_frame[int(center_y), int(
                            center_x)].astype(float)
                        distance = depth * cam.depth_scale
                    else:
                        # No valid distance found
                        distance = 0.0

                    #print(label + ' ' + str(self.truncate(distance, 2)) + 'm')

                    # Get translation vector relative to the camera frame
                    tvec = cam.deproject(
                        cam_intrinsics, center_x, center_y, distance)

                    # Create bounding box around object
                    frame = cv2.rectangle(
                        frame, (xmin, ymin), (xmax, ymax), color, 2)

                    # Put label and confidence on bounding box
                    frame = cv2.putText(frame, f'{label} P:({str(score)}) z: {str(self.truncate(tvec[2], 2))}', (
                        xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)

                    # Store log and send ZMQ message
                    if label == target_object:
                        self.logger.record_value([np.array(
                            [tvec[0], tvec[1], tvec[2], elapsed_time, score, label]), ])

                        msg = detection_msg_pb2.Detection()
                        msg.x = tvec[0]
                        msg.y = tvec[1]
                        msg.z = tvec[2]
                        msg.label = label
                        msg.confidence = score
                        serial_msg = msg.SerializeToString()

                old_frame_gray = frame_gray
                # Write resulting frame to output
                output.write(frame)
                if serial_msg is not None:
                    self.socket.send(serial_msg)
                else:
                    msg = detection_msg_pb2.Detection()
                    msg.x = 0.0
                    msg.y = 0.0
                    msg.z = 0.0
                    msg.label = 'Nothing'
                    msg.confidence = 0.0
                    serial_msg = msg.SerializeToString()
                    self.socket.send(serial_msg)

                # cv2.imwrite('pictures/frame_color.png', frame)
                frame_counter += 1
                elapsed_time = time.time() - starting_time

        except KeyboardInterrupt as e:
            output.release()
            cam.release()
            self.logger.export_to_csv(self.LOGFILE)


if __name__ == '__main__':
    det = Detector('weights/yolov5s.pt')
    det.detect_objects_cont("person")
