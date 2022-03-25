from tkinter import E, Frame
import cv2
import pyrealsense2 as rs
from realsense import RSCamera
import numpy as np
from elements.yolo import OBJ_DETECTION


WEIGHTS = 'weights/yolov5l.pt'

Object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                  'hair drier', 'toothbrush']

Object_colors = list(np.random.rand(80, 3)*255)
Object_detector = OBJ_DETECTION(WEIGHTS, Object_classes)

print(f'Using {WEIGHTS}')

cam = RSCamera()

output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 10, (cam.width, cam.height))

# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
#cap = cv2.VideoCapture(1)
try:
    #window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
    # Window
    while True:

        frame, depth = cam.get_frames()

        # detection process
        objs = Object_detector.detect(frame)

        # plotting
        for obj in objs:
            print(obj)
            label = obj['label']
            score = obj['score']
            [(xmin, ymin), (xmax, ymax)] = obj['bbox']
            color = Object_colors[Object_classes.index(label)]

            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            frame = cv2.putText(frame, f'{label} ({str(score)})', (
                xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)
            output.write(frame)
        # cv2.imshow("CSI Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    # cv2.destroyAllWindows()
    output.release()

except KeyboardInterrupt as e:
    output.release()
