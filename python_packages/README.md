# Python inference tool

Currently, supported models and inference frameworks:


* Image classification: based on https://github.com/lxztju/pytorch_classification.  support torch, onnxruntime and tensorrt inference  and multiple classification models.

* object detection: based on yolov5.  support torch, onnxruntime and tensorrt inference.

* Face detection and recognition: based on insightface. support torch, onnxruntime and tensorrt inference.



## Image classification
In the classification directory.

1. Download the training code to train the classification model.
```
git clone  https://github.com/lxztju/pytorch_classification.git
```

2. Modify the `classification.yaml` file
3. Run `export.py` to export the corresponding reasoning file
4. run `cls_infer.py`.


## Target detection
In the detection directory.

1. Download the yolov5 code to train a detection model. Then export the file in the corresponding format
2. Modify the `yolov5.yaml` file
3. run `yolov5_infer.py`.


## Face recognition
In face_recognition directory.

1. Download the insightface code and train the face recognition model.
2. Modify the `face.yaml `file
3. Run e`xport.py` to export the corresponding reasoning file
4. run `face_infer.py`.


>Note: The inference process of the above model only obtains the output results of different inference engines. The post-processing part of the model output results is not implemented. You can refer to the post-processing code which is implemented on the corresponding code repository.