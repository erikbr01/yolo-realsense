from tkinter import E, Frame
import cv2
import pyrealsense2 as rs
from realsense import RSCamera
import numpy as np
from elements.yolo import OBJ_DETECTION
from logger import Logger


class Detector:
    def __init__(self, weight_file) -> None:
        self.RECORD_COUNTER = self.get_record_counter('counter')
        self.OBJECT_LOG_NAME = 'bottle'
        self.LOGFILE = f"logs/records_{self.OBJECT_LOG_NAME}_{self.RECORD_COUNTER}.csv"
        self.WEIGHTS = weight_file

        self.object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                               'hair drier', 'toothbrush']

        self.object_colors = list(np.random.rand(80, 3)*255)

        self.logger = Logger()

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

    def detect_objects(self):

        object_detector = OBJ_DETECTION(self.WEIGHTS, self.object_classes)

        cam = RSCamera()

        output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 10, (cam.width, cam.height))

        try:
            while True:
                frame, depth = cam.get_frames()

                # detection process
                objs = object_detector.detect(frame)

                # plotting
                for obj in objs:
                    print(obj)
                    label = obj['label']
                    score = obj['score']
                    [(xmin, ymin), (xmax, ymax)] = obj['bbox']
                    color = self.object_colors[self.object_classes.index(
                        label)]

                    frame = cv2.rectangle(
                        frame, (xmin, ymin), (xmax, ymax), color, 2)
                    frame = cv2.putText(frame, f'{label} ({str(score)})', (
                        xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)
                    output.write(frame)

        except KeyboardInterrupt as e:
            output.release()
