import os
import sys

import torch
from pathlib import Path
import numpy as np
from collections import OrderedDict, namedtuple

from python_packages.tools import LOGGER


class MultiBackendInfer:
    def __init__(self, model_path, cpu_only=True):
        self.model_path = Path(model_path)
        self.cpu_only = cpu_only
        self.device = torch.device('cpu') if self.cpu_only else torch.device('cuda')
        print(self.device)
        self.dynamic = False
        
        self._init_backend()

    def _init_backend(self):
        if self.model_path.suffix  == '.torchscript':
            self._build_torchscript_infer_engine()
        elif self.model_path.suffix == '.onnx':
            
            self._build_onnx_infer_engine()
        elif self.model_path.suffix == '.engine':


            self._build_trt_infer_engine()
        elif self.model_path.suffix == '.openvino':
            self._build_openvino_infer_engine()
        else:
            raise ValueError(f'the model format of {self.model_path.suffix} is not support now.')

    def _build_torchscript_infer_engine(self):
        file = str(self.model_path)
        LOGGER.info(f"loadding {file} for torchscript inference...")
        self.torchscript_model = torch.jit.load(file, map_location=self.device)

    def _build_onnx_infer_engine(self):
        import onnxruntime
        file = str(self.model_path)
        LOGGER.info(f"loadding {file} for onnx runtime inference...")
        if self.cpu_only:
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = onnxruntime.InferenceSession(file, providers=providers)
        self.ort_input_names = [x.name for x in self.ort_session.get_inputs()]
        self.ort_output_names = [x.name for x in self.ort_session.get_outputs()]

    def _build_trt_infer_engine(self):
        try:
            import tensorrt as trt
        except Exception:
            raise ValueError("tensorrt import error! ")
        file = str(self.model_path)
        LOGGER.info(f"loadding {file} for  tensorrt engine inference...")
        if self.cpu_only:
            raise ValueError('tensorrt infer must use cuda, cpu only is not support.')

        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(file, 'rb') as f, trt.Runtime(logger) as runtime:
            self.trt_model = runtime.deserialize_cuda_engine(f.read())
        self.trt_context = self.trt_model.create_execution_context()
        self.trt_bindings = OrderedDict()
        self.trt_output_names = []
        self.trt_input_names = []
        for i in range(self.trt_model.num_bindings):
            name = self.trt_model.get_binding_name(i)
            dtype = trt.nptype(self.trt_model.get_binding_dtype(i))
            if self.trt_model.binding_is_input(i):
                if -1 in tuple(self.trt_model.get_binding_shape(i)):
                    self.dynamic=True
                    self.trt_context.set_binding_shape(i, tuple(self.trt_model.get_profile_shape(0, i)[2]))
                self.trt_input_names.append(name)
            else:
                self.trt_output_names.append(name)
            shape = tuple(self.trt_context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.trt_bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.trt_bindings.items())


    def _build_openvino_infer_engine(self):
        file = str(self.model_path)
        LOGGER.info(f"loadding {file} for openvino engine inference...")
        pass



    def __call__(self, img):
        if self.model_path.suffix  == '.torchscript':
            result = self._run_torchscript_infer_engine(img)

        elif self.model_path.suffix == '.onnx':
            result = self._run_onnx_infer_engine(img)

        elif self.model_path.suffix == '.engine':
            result = self._run_trt_infer_engine(img)

        elif self.model_path.suffix == '.openvino':
            result = self._run_openvino_infer_engine(img)

        else:
            raise ValueError(f'the model format of {self.model_path.suffix} is not support now.')
        return result


    def _run_torchscript_infer_engine(self, img):
        LOGGER.info("Torchscript infer process is running...")
        img = img.to(self.device)
        result = self.torchscript_model(img)
        result = result.detach() if result.requires_grad else result
        return result


    def _run_onnx_infer_engine(self, img):
        LOGGER.info("Onnx runtime infer process is running...")
        img = img.cpu().numpy()
        result = self.ort_session.run(self.ort_output_names, {self.ort_input_names[0]:img})
        return result


    def _run_trt_infer_engine(self, img):
        LOGGER.info("Tensorrt infer process is running...")
        img = img.to(self.device)

        trt_input_name = self.trt_input_names[0]
        print('trt_input_name: ', self.trt_input_names, self.trt_output_names)
        if self.dynamic and img.shape != self.trt_bindings[trt_input_name]._replace(shape=img.shape):
            print('******')
            i = self.trt_model.get_binding_index(trt_input_name)
            self.trt_context.set_binding_shape(i, img.shape)  # reshape if dynamic
            self.trt_bindings[trt_input_name] = self.trt_bindings[trt_input_name]._replace(shape=img.shape)
            for name in self.trt_output_names:
                i = self.trt_model.get_binding_index(name)
                self.trt_bindings[name].data.resize_(tuple(self.trt_context.get_binding_shape(i)))
        s = self.trt_bindings[trt_input_name].shape
        assert img.shape == s, f"input size {img.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        print('img.data_ptr(): ', img.data_ptr())
        self.binding_addrs[trt_input_name] = int(img.data_ptr())
        print('self.trt_bindings: ', self.trt_bindings)
        self.trt_context.execute_v2(list(self.binding_addrs.values()))
        print(self.trt_bindings)
        result = [self.trt_bindings[x].data for x in sorted(self.trt_output_names)]
        return result


    def _run_openvino_infer_engine(self, img):
        LOGGER.info("Openvino infer process is running...")
        result = ''
        return result