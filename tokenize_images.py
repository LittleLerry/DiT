from diffusers import AutoencoderKL
import torch
from torchvision import transforms
import io
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.multiprocessing as mp
from infer import save_tensor_to_image
import argparse
import glob
from utils import AIGenerated
import multiprocessing
import os

image_pt_suffix = ".image.pt"
label_pt_suffix = ".label.pt"

@AIGenerated
class parquet_image_dataset(Dataset):
    def __init__(self, dataframe, data_name, label_name, transform):
        self.df = dataframe
        self.data_name = data_name
        self.label_name = label_name
        self.transform = transform
                
    def __len__(self):
        return len(self.df)
                
    def __getitem__(self, idx):
        img_bytes = self.df.iloc[idx][self.data_name]["bytes"] # bytes data
        label = self.df.iloc[idx][self.label_name]

        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        return (self.transform(img), torch.tensor([label], dtype=torch.int16)) # ((B,C,H,W),(,))

def tokenize(tokenize_model, image): # image is a tensor with (B, C, H, W)
    posterior = tokenize_model.encode(image).latent_dist
    latents = posterior.sample()
    return latents

def detokenize(tokenize_model, latents):
    return tokenize_model.decode(latents).sample

def tokenize_task(rank, file_list, tokenizer_model_path, width):
    file_list = file_list[rank]

    assert torch.cuda.is_available()
    device = f"cuda:{rank}"
    model = AutoencoderKL.from_pretrained(tokenizer_model_path).to(device) # 32 precision
    preprocess = transforms.Compose([
        transforms.CenterCrop(width),
        transforms.Resize((width, width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    for file in file_list:
        latents = []
        labels = []

        dataset = parquet_image_dataset(pd.read_parquet(file), "image", "label", preprocess)
        dataloader = DataLoader(dataset, batch_size=256, num_workers=2, shuffle=False) # batch_size=256 will use ~70 GB memory per device
        
        model.eval()
        with torch.inference_mode():
            for (image, label) in dataloader:
                input = image.to(device)
                latents.append(tokenize(model, input).cpu())
                labels.append(label)
        
        final_latents = torch.cat(latents, dim=0)
        final_labels = torch.cat(labels, dim=0)

        image_save_path = file.replace('.parquet', image_pt_suffix)
        label_save_path = file.replace('.parquet', label_pt_suffix)

        torch.save(final_latents, image_save_path)
        torch.save(final_labels, label_save_path)
        print(f"rank:{rank} finished processing {file}")

        del latents, labels, final_latents, final_labels, dataset

        torch.cuda.empty_cache()
    print(f"rank{rank} done.")
    return

def _debug():
    # debug only
    # tokenize_task(0, ["/mnt/GPU_10T/data/zzx/data/train-00171-of-00294.parquet"], "/home/zzx/cs/DiT/tokenizer", 256)
    file = "/mnt/GPU_10T/data/zzx/data/train-00172-of-00294.parquet"
    tokenizer_model_path = "/mnt/GPU_10T/data/zzx/DiT/tokenizer"
    width = 256
    device = "cuda:0"
    preprocess = transforms.Compose([
        transforms.CenterCrop(width),
        transforms.Resize((width, width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    batch_size = 4
    dataset = parquet_image_dataset(pd.read_parquet(file), "image", "label", preprocess)
    model = AutoencoderKL.from_pretrained(tokenizer_model_path).to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    
    latents = []

    model.eval()
    with torch.inference_mode():
        for (image, label) in dataloader:
            for i in range(image.shape[0]):
                save_tensor_to_image(image[i], ".", f"{i}.png")
            print(f"input shape:{image.shape}")
            input = image.to(device)
            latents.append(tokenize(model, input).cpu())
            print(f"encoded shape:{latents[0].shape}")
            _d = detokenize(model, latents[0].to(device))
            print(f"decoded shape:{_d.shape}")

            for i in range(_d.shape[0]):
                save_tensor_to_image(_d[i], ".", f"d{i}.png")
            break

@AIGenerated
def split_list_into_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    # Calculate chunk size and remainder
    k, m = divmod(len(lst), n)
    # Create chunks: first m chunks have size k+1, remaining have size k
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tokenize images')
    parser.add_argument('--num_processes', type=int, default= 8)
    parser.add_argument('--data', type=str, default= "/mnt/GPU_10T/data/zzx/data")
    parser.add_argument('--tokenizer', type=str, default= "/mnt/GPU_10T/data/zzx/DiT/tokenizer")
    parser.add_argument('--width', type=int, default= 256)
    parser.add_argument('--debug', type=bool, default= False)

    parser.add_argument('--packed_image_tensor_path', type=str, default= "./data")
    parser.add_argument('--packed_label_tensor_path', type=str, default= "./data")

    args = parser.parse_args()

    debug = args.debug
    if debug:
        _debug()
        exit(0)

    num_processes = args.num_processes

    parquet_files = glob.glob(args.data + "/*.parquet")

    # parquet_files = parquet_files[:8]

    tasks = split_list_into_chunks(parquet_files, num_processes)

    # args = [(tasks[rank], args.tokenizer, args.width) for rank in range(8)]


    print(f"launching {num_processes} processes")

    # RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     pool.starmap(tokenize_task, args)
    
    mp.spawn(fn=tokenize_task, args=(tasks, args.tokenizer, args.width), nprocs=num_processes, join=True) # To be trained on 8*H800

    
    print("Main process starts to pack images and labels")

    images = []
    labels = []

    image_pt_files = glob.glob(args.data + "/*" + image_pt_suffix)
    label_pt_files = glob.glob(args.data + "/*" + label_pt_suffix)

    for i in image_pt_files:
        images.append(torch.load(i, weights_only=True))
    for l in label_pt_files:
        labels.append(torch.load(l, weights_only=True))
    print("stacking images and labels")
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    print(f"saving images {images.shape} and labels {labels.shape}")
    torch.save(images, os.path.join(args.packed_image_tensor_path, "i.pt"))
    torch.save(labels, os.path.join(args.packed_label_tensor_path, "l.pt"))