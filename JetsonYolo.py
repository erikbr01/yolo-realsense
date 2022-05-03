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

OBJ_THRESH = .6
P_THRESH = .3
NMS_THRESH = .5


class Detector:
    def __init__(self, weight_file) -> None:
        self.RECORD_COUNTER = self.get_record_counter('counter')
        self.LOG_NAME = 'testing'

        self.LOGFILE = f"logs/{self.LOG_NAME}_{self.RECORD_COUNTER}.csv"
        self.VIDEO_OUT_FILE = f'videos/{self.LOG_NAME}_{self.RECORD_COUNTER}.avi'
        self.WEIGHTS = weight_file

        self.ZMQ_SOCKET_ADDR = "tcp://localhost:5555"

        self.object_classes = coco

        self.object_colors = list(np.random.rand(80, 3)*255)

        self.logger = Logger()
        self.records = np.empty((0, self.logger.cols))

        # Initalize ZMQ connection
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
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

        starting_time = time.time()
        frame_counter = 0
        elapsed_time = 0

        try:
            while True:
                # To sync the frame capture with the motion capture data, we only capture frames when receiving something
                frame, depth_frame = cam.get_rs_color_aligned_frames()

                # We aligh depth to color, so we should use the color frame intrinsics
                cam_intrinsics = frame.profile.as_video_stream_profile().intrinsics

                # Now get raw frama data as tensors
                frame = np.asanyarray(
                    frame.get_data())
                depth_frame = np.asanyarray(
                    depth_frame.get_data())

                # Detection every 5 frames, otherwise tracking
                perform_detection = frame_counter % 5 == 0
                # perform_detection = True
                if perform_detection:
                    print("YOLO DETECTION")
                    objs = object_detector.detect(frame)
                    mtracker = cv2.legacy.MultiTracker_create()
                    for obj in objs:
                        [(xmin, ymin), (xmax, ymax)] = obj['bbox']
                        center_x = (xmax - xmin)/2 + xmin - 10
                        center_y = (ymax - ymin)/2 + ymin - 10
                        # mtracker.add(cv2.legacy.TrackerMedianFlow_create(),
                        #              frame, (xmin, ymin, w, h))
                        mtracker.add(cv2.legacy.TrackerMedianFlow_create(),
                                     frame, (center_x, center_y, 20, 20))
                else:
                    print("MTRACKER TRACKING")
                    is_tracking, bboxes = mtracker.update(frame)
                    if is_tracking:
                        for i, bbox in enumerate(bboxes):
                            xmin, ymin, w, h = [int(val) for val in bbox]
                            xmax = xmin + w
                            ymax = ymin + h
                            objs[i]['bbox'] = [(xmin, ymin), (xmax, ymax)]

                # plotting
                for obj in objs:
                    print(obj)
                    label = obj['label']
                    score = obj['score']
                    [(xmin, ymin), (xmax, ymax)] = obj['bbox']
                    color = self.object_colors[self.object_classes.index(
                        label)]

                    center_x = (xmax - xmin)/2 + xmin
                    center_y = (ymax - ymin)/2 + ymin

                    depth = depth_frame[int(center_y), int(
                        center_x)].astype(float)
                    distance = depth * cam.depth_scale

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

                # Write resulting frame to output
                output.write(frame)
                frame_counter += 1
                elapsed_time = time.time() - starting_time

        except KeyboardInterrupt as e:
            output.release()
            cam.release()
            self.logger.export_to_csv(self.LOGFILE)


if __name__ == '__main__':
    det = Detector('weights/yolov5s.pt')
    det.detect_objects_cont("person")
