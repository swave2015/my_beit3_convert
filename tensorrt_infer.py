from tensorrt_utils import TensorRTModel
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from PIL import Image
from transformers import XLMRobertaTokenizer
import torch
import cv2
import time

test_img_path = "/data/caoxh/code/my_beit3_convert/test_imgs/frame_2.png"
img_trt_model_path = "/data/caoxh/code/my_beit3_convert/tensorrt_models/beit3_retrival_coco_img_fp16_mine.trt"
input_size = 384
input_img = cv2.imread(test_img_path)
transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
input_img = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
input_img = transform(input_img).unsqueeze(0).cuda()
print(f"input_img_shape: {input_img.shape}")
img_trt_model = TensorRTModel(img_trt_model_path)

text_trt_model_path = "./tensorrt_models/beit3_retrival_coco_text_fp16_mine.trt"
text_trt_model = TensorRTModel(text_trt_model_path)

for i in range(100):
    start_time = time.time()
    image_feature = img_trt_model(inputs={'input_img': input_img})['img_feature']
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000
    print(f"img execution time: {elapsed_time:.5f} ms")
print(image_feature)


test_prompt = ['a picture of a dog', 'a picture of a cat']

tokenizer = XLMRobertaTokenizer('/data/caoxh/code/my_beit3_convert/model_weights/beit3.spm')
bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id
max_len = 64

def get_tokens(text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    max_len = 64
    token_ids = [bos_token_id] + token_ids[:] + [eos_token_id]
    num_tokens = len(token_ids)
    token_ids = token_ids + [pad_token_id] * (max_len - num_tokens)
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0)
    num_tokens = len(token_ids)
    token_ids = token_ids + [pad_token_id] * (max_len - num_tokens)
    padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
    padding_mask_tensor = torch.tensor(padding_mask).unsqueeze(0)

    return token_ids_tensor, padding_mask_tensor



token_ids_tensor0,  padding_mask_tensor0 = get_tokens(test_prompt[0])
token_ids_tensor0 = token_ids_tensor0.cuda()
padding_mask_tensor0 = padding_mask_tensor0.cuda()
text_feature0 = text_trt_model(inputs={'input_text': token_ids_tensor0, 'input_mask': padding_mask_tensor0})['text_feature']
print(text_feature0)

token_ids_tensor1,  padding_mask_tensor1 = get_tokens(test_prompt[1])
token_ids_tensor1 = token_ids_tensor1.cuda()
padding_mask_tensor1 = padding_mask_tensor1.cuda()
for i in range(100):
    start_time = time.time()
    text_feature1 = text_trt_model(inputs={'input_text': token_ids_tensor1, 'input_mask': padding_mask_tensor1})['text_feature']
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000
    print(f"text execution time: {elapsed_time:.5f} ms")
print(text_feature1)

print(image_feature @ text_feature0.t())
print(image_feature @ text_feature1.t())