import os
import tensorrt as trt
from tensorrt.tensorrt import Logger, Runtime
from tensorrt_utils import TensorRTShape, my_build_engine

if __name__ == '__main__':
    trt_logger: Logger = trt.Logger(trt.Logger.INFO)
    runtime: Runtime = trt.Runtime(trt_logger)
    trt.init_libnvinfer_plugins(trt_logger, '')
    input_onnx_path = '/data/caoxh/code/my_beit3_convert/model_weights/yolov8_bigdet_epoch60_last.onnx'
    engine = my_build_engine(
                runtime=runtime,
                onnx_file_path=input_onnx_path,
                logger=trt_logger,
                workspace_size=10000 * 1024 * 1024,
                fp16=True,
                int8=False
            )
    vision_fp16_trt_path = '/data/caoxh/code/my_beit3_convert/tensorrt_models/yolov8_bigdet_epoch60_last.trt'
    with open(vision_fp16_trt_path, 'wb') as f:
                f.write(bytearray(engine.serialize()))
