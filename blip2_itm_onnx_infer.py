from PIL import Image
import onnxruntime as ort
from transformers import BertTokenizer
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = 'cpu'
txt = "an image of a white cat"
raw_image = Image.open("./test_imgs/00000004.jpg").convert("RGB")


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

raw_image = Image.open("./test_imgs/00000004.jpg").convert("RGB")
img = img_transform(raw_image).unsqueeze(0).to(device)
print(f"img_shape: {img.shape}")

txt = "an image of a white cat"
text_token = tokenizer(txt, 
                            padding="max_length",
                            truncation=True, 
                            max_length=32,
                            return_tensors="pt").to(device)


onnx_path = '/data/caoxh/code/my_beit3_convert/blip2_onnx_full_model_export/blip2_itm_constant.onnx'
ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

inputs = {
    'input_img': img.cpu().numpy(),
    'token_input_ids': text_token.input_ids.cpu().numpy(),
    'token_attention_mask': text_token.attention_mask.cpu().numpy()

}
ort_outs = ort_session.run(None, inputs)
print(f"img_ort_outs: {ort_outs[0]}")