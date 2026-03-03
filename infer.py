import torch
import os
from PIL import Image
import numpy as np

def sampler(ddp_model, samplings, infer_steps, guidance_scale, label, empty_label, continue_train = False) -> torch.Tensor:
    ddp_model.eval()
    with torch.inference_mode():
        t = torch.zeros(size=samplings.shape[:-3], device=samplings.device) # timestamps
        h = 1 / infer_steps
        for _ in range(infer_steps):
            u_empty = ddp_model(samplings, t, empty_label) # (*, C, H, W)
            u = ddp_model(samplings, t, label) # (*, C, H, W)
            u_hat = (1 - guidance_scale) * u_empty + guidance_scale * u
            samplings = samplings + h * u_hat
            t = t + h
        return samplings
    if continue_train:
        ddp_model.train()

def save_tensor_to_image(tensor, save_path, image_name):
    os.makedirs(save_path, exist_ok=True)

    tensor = tensor.detach().cpu()
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor.permute(1, 2, 0)

    numpy_img = tensor.numpy()
    numpy_img = (numpy_img * 255).astype(np.uint8)

    pil_img = Image.fromarray(numpy_img, mode='RGB')
    full_path = os.path.join(save_path, image_name)

    pil_img.save(full_path)

def convert_image_to_tensor(image):
    pass