import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # DL_deploy root directory
# print([x for x in FILE.parents])
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import torch
from tools import LOGGER, load_yaml_file
import numpy as np
from insightface.detection.scrfd.mmdet.core import (build_model_from_cfg, generate_inputs_and_wrap_model,
                        preprocess_example_input)

class ScrfdExportor:
    def __init__(self, config, cpu_only=True):
        self.config = config
        self.model_config_path = config['face_detection']['model_config_path']
        self.model_path = Path(config['face_detection']['model_path'])
        self.input_dims = config['face_detection']['input_dims']
        self.input_image = config['face_detection']['input_image']
        self.mean = config['face_detection']['mean']
        self.std = config['face_detection']['std']

        self.input_config = {
            'input_shape': self.input_dims,
            'input_path': self.input_image,
            'normalize_cfg': {
                'mean': self.mean,
                'std': self.std
            }
        }
        self.cpu_only = cpu_only
        if self.cpu_only:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')

        self._load_torch_model()

    def _load_torch_model(self):
        checkpoint_path = self.model_path
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'optimizer' in checkpoint:
            del checkpoint['optimizer']
            tmp_ckpt_file = self.model_path + "_slim.pth"
            torch.save(checkpoint, tmp_ckpt_file)
            print('remove optimizer params and save to', tmp_ckpt_file)
            checkpoint_path = tmp_ckpt_file

        self.scrfd_torch_model, self.dummy_input = generate_inputs_and_wrap_model(
            self.model_config_path, checkpoint_path, self.input_config)
        return self.scrfd_torch_model


    def _export_torchscript(self):
        target_path = self.model_path.with_suffix('.torchscript')
        if target_path.exists():
            LOGGER.info('torchscript is existed, exit. ')

        self.scrfd_torchscript_model = torch.jit.trace(
            self.scrfd_torch_model,
            (self.dummy_input,)
        )
        self.scrfd_torchscript_model.save(target_path)
        LOGGER.info(f"torchscript is exported and saved to {target_path}")

    def _export_onnx(self):
        target_path = self.model_path.with_suffix('.onnx')
        if target_path.exists():
            LOGGER.info('onnx file is existed, exit. ')

        input_names = ['input.1']
        output_names = ['score_8', 'score_16', 'score_32',
                        'bbox_8', 'bbox_16', 'bbox_32',
                        ]
        if 'stride_kps' in str(self.scrfd_torch_model):
            output_names += ['kps_8', 'kps_16', 'kps_32']


        dynamic_axes = {out: {0: '?', 1: '?'} for out in output_names}
        dynamic_axes[input_names[0]] = {
            0: '?',
            2: '?',
            3: '?'
        }
        torch.onnx.export(
            self.scrfd_torch_model,
            (self.dummy_input,),
            target_path,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11
        )
        LOGGER.info(f"onnx file is exported and saved to {target_path}")


    def _export_trt(self):
        target_path = self.model_path.with_suffix('.engine')
        if target_path.exists():
            LOGGER.info('tensorrt engine file is existed, exit. ')

        self._export_onnx()
        try:
            import tensorrt as trt
        except Exception:
            raise ValueError("tensorrt import error! ")

        onnx_path = target_path.with_suffix('.onnx')

        assert onnx_path.exists(), f'failed to export ONNX file: {onnx_path}'

        logger = trt.Logger(trt.Logger.INFO)  # 创建trt logger

        builder = trt.Builder(logger)  # 创建builder
        workspace = config['workspace']  # 工作空间大小，4g
        trt_config = builder.create_builder_config()  # 创建config
        trt_config.max_workspace_size = workspace * 1 << 30
        # 隐式batch中，tensor中没有batch维度的信息，并且所有维度必须是常数。
        # tensorrt保留隐式batch是为了向后兼容。因此新代码不推荐使用隐式batch。
        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)  # 创建模型
        parser = trt.OnnxParser(network, logger)  ## onnx parser
        if not parser.parse_from_file(str(onnx_path)):
            raise RuntimeError(f'failed to load ONNX file: {onnx_path}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]

        imgsz = self.config['face_detection']['input_dims']
        im = torch.zeros(*imgsz).to(self.device)

        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
        trt_config.add_optimization_profile(profile)

        with builder.build_engine(network, trt_config) as engine, open(target_path, 'wb') as t:
            t.write(engine.serialize())

        LOGGER.info(f"tensorrt engine file is exported and saved to {target_path}")

    def export(self):
        export_type = config['convert_type']
        LOGGER.info(f'export type: {export_type}')
        if "onnx" in export_type:
            LOGGER.info("starting convert onnx")
            self._export_onnx()
        if 'tensorrt' in export_type:
            LOGGER.info("starting convert trt model")
            self._export_trt()
        if 'torchscript' in export_type:
            LOGGER.info("starting convert torchscript model")
            self._export_torchscript()




class ArcfaceExportor:
    def __init__(self, config):
        self.config = config
        self.model_path = Path(config['face_recognition']['model_path'])

        if config['cpu_only']:
            self.device=torch.device('cpu')
        else:
            self.device= torch.device('cuda')

        self._load_torch_model()
        self.dummy_input = self._init_dummy_input()

    def _load_torch_model(self):
        from models.recognition.arcface_torch.backbones import get_model
        self.torch_model = get_model(self.config['face_recognition']['model_name'], dropout=0.0, fp16=False, num_features=512)
        weight = torch.load(self.model_path, map_location=self.device)
        self.torch_model.load_state_dict(weight, strict=True)
        self.torch_model.eval()


    def _init_dummy_input(self):
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = img.astype(np.float32)
        img = (img / 255. - 0.5) / 0.5  # torch style norm
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        return img.to(self.device)

    def _export_torchscript(self):
        target_path = self.model_path.with_suffix('.torchscript')
        if target_path.exists():
            LOGGER.info('torchscript is existed, exit. ')
        traced_script_module = torch.jit.trace(self.torch_model, self.dummy_input)
        traced_script_module.save(target_path)
        LOGGER.info(f"torchscript is exported and saved to {target_path}")


    def _export_onnx(self):
        target_path = self.model_path.with_suffix('.onnx')
        if target_path.exists():
            LOGGER.info('onnx file is existed, exit. ')
        torch.onnx.export(
            self.torch_model,
            self.dummy_input,
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
        LOGGER.info(f"onnx file is exported and saved to {target_path}")


    def _export_trt(self):
        target_path = self.model_path.with_suffix('.engine')
        if target_path.exists():
            LOGGER.info('tensorrt engine file is existed, exit. ')

        self._export_onnx()
        try:
            import tensorrt as trt
        except Exception:
            raise ValueError("tensorrt import error! ")

        onnx_path = target_path.with_suffix('.onnx')

        assert onnx_path.exists(), f'failed to export ONNX file: {onnx_path}'

        logger = trt.Logger(trt.Logger.INFO)  # 创建trt logger

        builder = trt.Builder(logger) # 创建builder
        workspace = config['workspace'] # 工作空间大小，4g
        trt_config = builder.create_builder_config()  # 创建config
        trt_config.max_workspace_size = workspace * 1 << 30
        # 隐式batch中，tensor中没有batch维度的信息，并且所有维度必须是常数。
        # tensorrt保留隐式batch是为了向后兼容。因此新代码不推荐使用隐式batch。
        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag) # 创建模型
        parser = trt.OnnxParser(network, logger) ## onnx parser
        if not parser.parse_from_file(str(onnx_path)):
            raise RuntimeError(f'failed to load ONNX file: {onnx_path}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]

        imgsz = self.config['face_recognition']['input_dims']
        im = torch.zeros(*imgsz).to(self.device)

        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
        trt_config.add_optimization_profile(profile)

        with builder.build_engine(network, trt_config) as engine, open(target_path, 'wb') as t:
            t.write(engine.serialize())

        LOGGER.info(f"tensorrt engine file is exported and saved to {target_path}")

    def export(self):
        export_type = config['convert_type']
        LOGGER.info(f'export type: {export_type}')
        if "onnx" in export_type:
            LOGGER.info("starting convert onnx")
            self._export_onnx()
        if 'tensorrt' in export_type:
            LOGGER.info("starting convert trt model")
            self._export_trt()
        if 'torchscript' in export_type:
            LOGGER.info("starting convert torchscript model")
            self._export_torchscript()



if __name__ == '__main__':
    config_file = './face.yaml'
    config = load_yaml_file(config_file)
    # arcface_exporter = ArcfaceExportor(config)
    # arcface_exporter.export()
    scrfd_exporter = ScrfdExportor(config)
    scrfd_exporter.export()
