import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # DL_deploy root directory
# print([x for x in FILE.parents])
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import torch
import torch.onnx
# import onnx
# import onnxruntime
from PIL import Image
import numpy as np

from models.cls_models import ClsModel
from models.dataset import val_transform

from python_packages.tools.load_yamls import load_yaml_file


def img_process(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = val_transform(size=config['data_params']['input_dims'][2])(img).unsqueeze(0)
    return img_tensor


def load_torch_model(config):
    torch_model = ClsModel(model_name=config['model_params']['name'], num_classes=config['model_params']['num_classes'], is_pretrained=False)
    sd = torch.load(config['model_params']['model_path'], map_location='cpu')
    from collections import OrderedDict
    new_sd = OrderedDict()
    for k, v in sd.items():
        if 'base_model' not in k:
            new_sd['base_model.' + k] = v
        else:
            new_sd[k] = v
    torch_model.load_state_dict(new_sd)
    torch_model.eval()
    return torch_model


def torch_model_infer(torch_model, img_tensor):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    torch_model.to(device)

    with torch.no_grad():
        img_tensor = img_tensor.to(device)

        output = torch_model(img_tensor)
        torch_result = torch.softmax(output, dim=1)
    return torch_result

def export_torchscript(config, verify):

    target_path = config['model_params']['model_path'].replace('pth', 'torchscript')
    if os.path.exists(target_path):
        print('torchscript has been convert completely.')
        return

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    torch_model = load_torch_model(config)
    img_path = config['data_params']['input_image']

    img_tensor = img_process(img_path)
    traced_script_module = torch.jit.trace(torch_model, img_tensor)
    traced_script_module.save(target_path)

    if verify:
        img_tensor = img_tensor.to(device)
        torch_result = torch_model_infer(torch_model, img_tensor)
        torch_result = torch_result.detach() if torch_result.requires_grad else torch_result
        torchscript_model = torch.jit.load(target_path).to(device)
        import time
        s1 = time.time()
        for _ in range(1000):
            torchscript_result = torchscript_model(img_tensor)
        s2 = time.time()
        print("torch_script_time:", 1/((s2 - s1)/1000))
        torchscript_result = torch.softmax(torchscript_result, dim=1)
        torchscript_result = torchscript_result.detach() if torchscript_result.requires_grad else torchscript_result
        #  print("torchscript_result: ", torchscript_result)
        #  print("torch_result: ", torch_result)
        #  print(np.max(torchscript_result.cpu().numpy(), axis=1), np.max(torch_result.cpu().numpy(), axis=1))
        # compare torchscript and PyTorch results
        np.testing.assert_allclose(torch_result.cpu().numpy(), torchscript_result.cpu().numpy(), rtol=1e-03, atol=1e-05)

        print("Exported model has been tested with torchscript, and the result looks good!")


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


def onnx_verify(torch_model, onnx_path):
    from python_packages.common.multibackend_infer import MultiBackendInfer
    onnx_classifier = MultiBackendInfer(onnx_path, cpu_only=False)

    import time
    img_path = config['data_params']['input_image']
    img_tensor = img_process(img_path)
    s1 = time.time()
    for _ in range(1000):
        onnx_result = onnx_classifier(img_tensor)[0]
        onnx_result = softmax(onnx_result)
    s2 = time.time()
    print("onnx_time:", 1/((s2 - s1)/1000))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    torch_model.to(device)

    with torch.no_grad():
        img_tensor = img_tensor.to(device)

        s1 = time.time()
        for _ in range(1000):
            output = torch_model(img_tensor)

        s2 = time.time()
        print("torch_time:", 1/((s2 - s1)/1000))
        torch_result = torch.softmax(output, dim=1).cpu()
    #  print("onnx_result: ", onnx_result)
    #  print("torch_result: ", torch_result)
    #  print(np.max(onnx_result, axis=1), np.max(torch_result.numpy(), axis=1))
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(torch_result.cpu().numpy(), onnx_result, rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")





def export_onnx(config, verify):
    target_path = config['model_params']['model_path'].replace('pth', 'onnx')
    if os.path.exists(target_path):
        print('onnx has been convert completely.')
        return
    torch_model = load_torch_model(config)

    dummy_input = torch.randn(config['data_params']['input_dims'])

    # print(target_path, os.path.dirname(target_path))
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    torch.onnx.export(
        torch_model,
        dummy_input,
        target_path,
        export_params=True,
        input_names=["input_image"],
        output_names=["model_output"],
        dynamic_axes={
            "input_image": {0: "batch"},
            "model_output": {0: "batch"}
                        },
        opset_version=11
    )

    if verify:
        onnx_verify(torch_model, target_path)


def export_trt(config, verify):

    export_onnx(config, False)
    file = Path(config['model_params']['model_path'])
    onnx_path = config['model_params']['model_path'].replace('pt', 'onnx')
    try:
        import tensorrt as trt
    except Exception:
        raise ValueError("tensorrt import error! ")

    onnx_path = file.with_suffix('.onnx')

    assert onnx_path.exists(), f'failed to export ONNX file: {onnx_path}'
    trt_path = file.with_suffix('.engine')  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(logger)
    workspace = config['workspace']
    trt_config = builder.create_builder_config()
    trt_config.max_workspace_size = workspace * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError(f'failed to load ONNX file: {onnx_path}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    imgsz = config['data_params']['input_dims']
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    im = torch.zeros(*imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection


    profile = builder.create_optimization_profile()
    for inp in inputs:
        profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
    trt_config.add_optimization_profile(profile)

    with builder.build_engine(network, trt_config) as engine, open(trt_path, 'wb') as t:
        t.write(engine.serialize())


def main(config):
    export_type = config['convert_type']
    print(export_type)
    verify = config['verify']
    if "onnx" in export_type:
        print("starting convert onnx")
        export_onnx(config, verify)
    if 'tensorrt' in export_type:
        print("starting convert trt model")
        export_trt(config, verify)
    if 'torchscript' in export_type:
        print("starting convert torchscript model")
        export_torchscript(config, verify)
    #  else:
        #  raise ValueError(f"The export_type of {export_type} is not support now.")



if __name__ == '__main__':
    config_file = './classification.yaml'
    config = load_yaml_file(config_file)
    main(config)
