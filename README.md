# DL_deploy
A deep learning model deployment toolset. Support Multiple frames , such astorch, onnxruntime, TensorRT.


## Introduction
This project is an open source computer vision deployment toolset.

It will realize the deployment of basic computer vision tasks, such as image classification, 
object detection, face recognition and other common tasks. It will support the process of 
model deployment  based on torch, onnx, tensor, openvino and other backends.
The model inference will be implemented by c++ and python.

## Recent update

* 20230113 update
    - support these inference frames of torchscript, onnxruntime, TensorRT of python.
    - support these inference frames of torchscript, onnxruntime of c++.
    -  The python package support multiple computer vision tasks such as image classification, target detection, face detection and recognition
        - The classification algorithm is based on: https://github.com/lxztju/pytorch_classification.git
        - the detection algorithm: yolov5
        - The face detection and recognition algorithm: insightface

    - The c++ only support image classification task now.

## Mainly requirements


* python package requirements

```text
torch==1.13.0
tensorrt==8.4.3.1
onnx==1.13.0
onnxruntime-gpu==1.13.1
```

* c++ requirements

```
libtorch==1.13.0
spdlog
opencv==3.4.16
onnxruntime==1.10.1
```


## Quick start

1. Compile the dependent third-party library to ` third_ Party ` and modify`CmakeLists.txt` .

2. compile

```
sh compile.sh
```

3. The binary file will be in `bin` and library files will be in `lib`.


* The inference process of python version：[python infer](python_packages/README.md)
