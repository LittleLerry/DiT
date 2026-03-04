from dit import dit
import torch
from train_dit import get_dataset_conf, get_general_conf
from dataset import tensor_image_dataset
from tokenize_images import detokenize, save_tensor_to_image
from diffusers import AutoencoderKL

ds = tensor_image_dataset(**get_dataset_conf())
device = "cuda:0"

vae = AutoencoderKL.from_pretrained(get_general_conf()["tokenizer_model_path"]).to(device)
vae.eval()

i , _ = ds.__getitem__(114514)

save_tensor_to_image(detokenize(vae, i.to(device)), ".","debug.png")
