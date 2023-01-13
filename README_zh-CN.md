# DL_deploy
一个pytorch深度学习模型部署工具箱，支持torch, onnxruntime,TensorRT多种推理方式。

## 介绍
这个项目是一个开源的计算机视觉部署工具箱。 它将实现基本计算机视觉任务的部署，例如，分类、目标检测、人脸识别和其他常见任务。它将支持基于torch 
、onnxruntime、TensorRT、openvino和其他后端的模型部署法。 支持c++和python两种语言的模型推理部署方式。

## 更新日志

* 20230113 update
    - 实现python语言的torchscript、onnxruntime、TensorRT的推理方式
    - 实现c++语言的torchscript、onnxruntime推理方式
    - python语言推理支持图像分类，目标检测，人脸检测和识别、
      - 其中，分类算法基于：https://github.com/lxztju/pytorch_classification.git
      - 检测算法： yolov5
      - 人脸检测识别： insightface
    - c++语言推理模型支持图像分类模型的推理。

## 主要依赖库


* python依赖

```text
torch==1.13.0
tensorrt==8.4.3.1
onnx==1.13.0
onnxruntime==1.13.1
```

* c++依赖的第三方库

```
libtorch==1.13.0
spdlog
opencv==3.4.16
onnxruntime==1.10.1
```



## 快速开始

1. 将所依赖的第三方库，编译到`third_party`文件夹中，同时修改`CmakeLists.txt`

2. 编译

```
sh compile.sh
```

3. 使用`bin`中生成的二进制文件和`lib`中的库文件


* python版本的任务推理过程：[python 推理使用流程](python_packages/README.md)
