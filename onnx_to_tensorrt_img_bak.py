import os
import tensorrt as trt
from tensorrt.tensorrt import Logger, Runtime
from tensorrt_utils import TensorRTShape, build_engine

if __name__ == '__main__':
    trt_logger: Logger = trt.Logger(trt.Logger.INFO)
    runtime: Runtime = trt.Runtime(trt_logger)
    trt.init_libnvinfer_plugins(trt_logger, '')
    #如果batch_size设置成其它的值会出现输入为nan的情况
    vision_input_shape = [TensorRTShape((1, 3, 384, 384),
                                        (1, 3, 384, 384),
                                        (1, 3, 384, 384), 'input_img')]
    print(f"vision_input_shape_name: {vision_input_shape[0].input_name}")
    
    input_vision_onnx_path = '/data/beit3_trt_convert/onnx_models/beit3_retrival_coco_img.onnx'
    # input_vision_onnx_path="/data/clip_trt/convert_output/ViT-L_14@336px.fp16.onnx"
    engine = build_engine(
                runtime=runtime,
                onnx_file_path=input_vision_onnx_path,
                logger=trt_logger,
                workspace_size=10000 * 1024 * 1024,
                fp16=True,
                int8=False,
                input_shapes=vision_input_shape
            )
    vision_fp32_trt_path = '/data/beit3_trt_convert/tensorrt_models/beit3_retrival_coco_img_fp16.trt'
    with open(vision_fp32_trt_path, 'wb') as f:
                f.write(bytearray(engine.serialize()))
