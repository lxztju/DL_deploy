

import numpy as np
import cv2
import torch
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from python_packages.common.multibackend_infer import MultiBackendInfer


class Yolov5Infer:
    def __init__(self, model_path, input_size=(640, 640), cpu_only=True):
        self.model_path = model_path
        self.cpu_only = cpu_only
        self.input_size = input_size
        self.infer_session = MultiBackendInfer(self.model_path, cpu_only=cpu_only)

    def _letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im


    def _img_process(self, img_path):
        img = cv2.imread(img_path)

        img = self._letterbox(img)
        img = img.transpose(2, 0, 1)
        # print(img.shape)
        img = torch.from_numpy(img).unsqueeze(0).float()
        return img

    def __call__(self, img_path):
        img = self._img_process(img_path)
        result = self.infer_session(img)
        return result







if __name__ == '__main__':

    model_path = './workspace/yolov5s.onnx'
    img_path = './workspace/samples/demo.jpg'
    yolov5_infer = Yolov5Infer(model_path=model_path)
    result = yolov5_infer(img_path)
    print([x.shape for x in result])
