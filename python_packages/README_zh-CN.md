# python推理工具使用文档、

目前支持的模型和推理框架：

* 图像分类：基于https://github.com/lxztju/pytorch_classification，支持多种分类模型的torch, onnxruntime和tensorrt推理

* 目标检测：基于yolov5,支持torch, onnxruntime和tensorrt推理


* 人脸检测识别：基于insightface，支持torch, onnxruntime和tensorrt推理


## 1. 图像分类

在classification目录下。

1. 下载分类模型训练代码，训练分类模型。git clone https://github.com/lxztju/pytorch_classification.git
2. 修改classification.yaml文件
3. 运行export.py，导出相应的推理文件
4. 执行cls_infer.py进行推理。


## 2. 目标检测


在detection目录下。

1. 下载yolov5代码训练模型。并导出相应格式的文件
2. 修改yolov5.yaml文件
3. 执行yolov5_ifer.py进行推理。


## 3. 人脸识别

在face_recognition目录下。

1. 下载insightface人脸识别代码，并训练人脸识别模型。
2. 修改face.yaml文件
3. 运行export.py，导出相应的推理文件
4. 执行face_infer.py进行推理。

> 注： 以上模型的推理过程，只是得到不同推理引擎的输出结果，对于模型输出结果的后处理部分，没有实现，可以自行参考相应代码仓库的后处理代码实现。

