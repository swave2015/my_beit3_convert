import os
import tensorrt as trt
from tensorrt.tensorrt import Logger, Runtime
from tensorrt_utils import TensorRTShape, my_build_engine

if __name__ == '__main__':
    trt_logger: Logger = trt.Logger(trt.Logger.INFO)
    runtime: Runtime = trt.Runtime(trt_logger)
    trt.init_libnvinfer_plugins(trt_logger, '')
    
    input_text_shape = [TensorRTShape((1, 64),
                                    (1, 64),
                                    (1, 64), 'input_text'),
                        TensorRTShape((1, 64),
                                    (1, 64),
                                    (1, 64), 'input_mask')]
       
    input_mask_shape = []
    
    input_text_onnx_path = '/data/beit3_trt_convert/onnx_models/beit3_retrival_coco_text_constant.onnx'
    # input_vision_onnx_path="/data/clip_trt/convert_output/ViT-L_14@336px.fp16.onnx"
    engine = my_build_engine(
                runtime=runtime,
                onnx_file_path=input_text_onnx_path,
                logger=trt_logger,
                workspace_size=10000 * 1024 * 1024,
                fp16=True,
                int8=False
            )
    text_fp32_trt_path = '/data/beit3_trt_convert/tensorrt_models/beit3_retrival_coco_text_fp16_mine.trt'
    with open(text_fp32_trt_path, 'wb') as f:
                f.write(bytearray(engine.serialize()))

   
    