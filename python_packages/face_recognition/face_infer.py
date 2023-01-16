

import numpy as np
import cv2
import torch
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from common.multibackend_infer import MultiBackendInfer


class ScrfdInfer:
    def __init__(self, model_path, input_size=(640, 640), cpu_only=True):
        self.model_path = model_path
        self.cpu_only = cpu_only
        self.input_size = input_size
        self.infer_session = MultiBackendInfer(self.model_path, cpu_only=cpu_only)

    def _img_process(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.input_size)
        img = img.astype(np.float32)
        img = cv2.dnn.blobFromImage(img, 1.0/128, self.input_size, (127.5, 127.5, 127.5), swapRB=True)
        img = torch.from_numpy(img).float()
        return img

    def __call__(self, img_path):
        img = self._img_process(img_path)
        result = self.infer_session(img)
        return result





class ArcfaceInfer:
    def __init__(self, model_path, input_size=(112, 112), cpu_only=True):
        self.model_path = model_path
        self.cpu_only = cpu_only
        self.input_size = input_size
        self.infer_session = MultiBackendInfer(self.model_path, cpu_only=cpu_only)

    def _img_process(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.input_size)
        img = img.astype(np.float32)
        img = (img / 255. - 0.5) / 0.5  # torch style norm
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        return img

    def __call__(self, img_path):
        img = self._img_process(img_path)
        result = self.infer_session(img)
        return result



if __name__ == '__main__':
    # model_path = './workspace/arcface_r50.torchscript'
    # img_path = './workspace/samples/demo_rec.jpg'
    # arc_infer = ArcfaceInfer(model_path=model_path)
    # result = arc_infer(img_path)
    # print(result[0].shape)

    model_path = './workspace/scrfd_2.5g_bnkps.onnx'
    img_path = './workspace/samples/demo_scrfd.jpg'
    scrfd_infer = ScrfdInfer(model_path=model_path)
    result = scrfd_infer(img_path)
    print([x.shape for x in result])
