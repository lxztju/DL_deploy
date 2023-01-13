import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # DL_deploy root directory
# print([x for x in FILE.parents])
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
from PIL import Image
from python_packages.tools import load_yaml_file
from models.dataset import val_transform

from python_packages.common.multibackend_infer import MultiBackendInfer
import numpy as np
import torch
import cv2
def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


# def img_process(img_path):
#     img = Image.open(img_path).convert('RGB')
#     img_tensor = val_transform(size=config['data_params']['input_dims'][2])(img).unsqueeze(0)
#     return img_tensor


def img_process(img_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # img = Image.open(img_path).convert('RGB')
    # img = img.resize((224, 224), Image.ANTIALIAS)
    #
    # # img = Resize((224, 224))(img)
    # img = np.array(img, dtype=np.float32)
    img = cv2.imread(img_path)
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
    img = img.div(255.)
    mean = torch.as_tensor(mean, dtype=torch.float32)
    mean = mean.view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=torch.float32)
    std = std.view(-1, 1, 1)
    img = img.sub_(mean).div_(std)

    return img.unsqueeze(0)


config_file = './classification.yaml'
config = load_yaml_file(config_file)
img_tensor = img_process(config['data_params']['input_image'])
infer_session = MultiBackendInfer(config['model_params']['model_path'].replace('.pth', '.torchscript'), cpu_only=True)
result = infer_session(img_tensor)
import torch
result = torch.softmax(result, 1)
print(result)

print(torch.max(result), torch.argmax(result))


# print(result[0].shape)
# result = softmax(result[0])
# print(result[0])
# print(np.max(result), np.argmax(result))

