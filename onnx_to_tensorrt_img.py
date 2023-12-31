import os
import tensorrt as trt
from tensorrt.tensorrt import Logger, Runtime
from tensorrt_utils import TensorRTShape, my_build_engine

if __name__ == '__main__':
    trt_logger: Logger = trt.Logger(trt.Logger.INFO)
    runtime: Runtime = trt.Runtime(trt_logger)
    trt.init_libnvinfer_plugins(trt_logger, '')
    
    # vision_input_shape = [TensorRTShape((1, 3, 384, 384),
    #                                     (2, 3, 384, 384),
    #                                     (3, 3, 384, 384), 'input_img')]
    # print(f"vision_input_shape_name: {vision_input_shape[0].input_name}")
    
    input_vision_onnx_path = '/data/caoxh/code/my_beit3_convert/model_weights/beit3_retrival_coco_img_constant.onnx'
    # input_vision_onnx_path="/data/clip_trt/convert_output/ViT-L_14@336px.fp16.onnx"
    engine = my_build_engine(
                runtime=runtime,
                onnx_file_path=input_vision_onnx_path,
                logger=trt_logger,
                workspace_size=10000 * 1024 * 1024,
                fp16=True,
                int8=False,
            )
    vision_fp32_trt_path = '/data/caoxh/code/my_beit3_convert/tensorrt_models/beit3_retrival_coco_img_fp16_mine.trt'
    with open(vision_fp32_trt_path, 'wb') as f:
                f.write(bytearray(engine.serialize()))
