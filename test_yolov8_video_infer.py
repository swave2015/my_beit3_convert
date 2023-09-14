from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

model = YOLO("./model_weights/yolov8_bigdet_epoch60_last.pt")

# im1 = Image.open("./test_imgs/387bdf8fa0c3f7273d11d4e2ad192d8e(1).jpg")

results = model.predict(source='./test_imgs', save=True, save_txt=True)
# image = Image.fromarray(np.array(results.orig_img))
# image.save("./results/output_rgb.jpg")