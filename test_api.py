from dit import dit
import torch
from train_dit import get_dataset_conf, get_general_conf
from dataset import tensor_image_dataset
from tokenize_images import detokenize, save_tensor_to_image
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

ds = tensor_image_dataset(**get_dataset_conf())
dl = DataLoader(ds, batch_size=5)
device = "cuda:0"

vae = AutoencoderKL.from_pretrained(get_general_conf()["tokenizer_model_path"]).to(device)
vae.eval()

for i , _ in dl:
    imgs = detokenize(vae, i.to(device))
    for i in range(imgs.shape[0]):
        save_tensor_to_image(imgs[i], ".",f"_{i}_debug.png")
    break
