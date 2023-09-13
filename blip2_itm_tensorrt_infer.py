from transformers import BertTokenizer
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from tensorrt_utils import TensorRTModel
import time

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="./huggingface_cache", truncation_side="right")
tokenizer.add_special_tokens({"bos_token": "[DEC]"})

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
normalize = transforms.Normalize(mean, std)
image_size = 224
img_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

raw_image = Image.open("./test_imgs/00008.jpg").convert("RGB")
img = img_transform(raw_image).unsqueeze(0).to(device)
print(f"img_shape: {img.shape}")

txt = "a bear that is walking"
text_token = tokenizer(txt, 
                            padding="max_length",
                            truncation=True, 
                            max_length=32,
                            return_tensors="pt").to(device)

model_path = "./blip2_trt_models/blip2_itm.trt"
model = TensorRTModel(model_path)

for i in range(100):
    start_time = time.time()
    itm_score = model(inputs={'input_img': img, 'token_input_ids': text_token.input_ids, 'token_attention_mask': text_token.attention_mask})['output']
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000
    print(f"blip2 itm execution time: {elapsed_time:.5f} ms")

print(f"itm_score: {itm_score}")