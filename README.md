# Yolo Object Detection and Localization with Intel Realsense Cameras

This repository is building up on the work of [this repository](https://github.com/amirhosseinh77/JetsonYolo) which provides code to run yolov5 using pytorch.

I am building up on this work to enable real time object detection *and localization* by deprojecting 2D points from the image to 3D points relative to the camera frame using an Intel Realsense D455 camera. This code will be used for aerial grasping for the [RAPTOR project](https://github.com/raptor-ethz). 


## Download Model
Select the desired model based on model size, required speed, and accuracy.
You can find available models [**here**](https://github.com/ultralytics/yolov5/releases) in the **Assets** section.
Download the model using the command below and move it to the **weights** folder.
```
$ cd weights
$ wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
```

##### PyTorch & torchvision
Yolov5 network model is implemented in the Pytorch framework.
PyTorch is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing.
Heres a complete guide to [**install PyTorch & torchvision**](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-9-0-now-available/72048) for Python on Jetson Development Kits

## Inference
Run ```JetsonYolo.py``` to detect objects with the camera.
```
$ python3 JetsonYolo.py
```
![Screenshot from 2021-07-07 03-25-48](https://user-images.githubusercontent.com/56114938/124771486-66ccaf00-df50-11eb-9d44-3f69d2a2a434.png)


