import torch
import torch.nn as nn
import random
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm
from infer import save_tensor_to_image, sampler
"""
A DiT trainer without EMA.
"""
def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def ddp_trainer(rank, conf, model_class, dataset_class, model_conf, dataset_conf):
    random.setstate(rank)
    # train conf
    world_size = conf["world_size"]
    batch_size = conf["batch_size"]
    lr = conf["lr"]
    epoch = conf["epoch"]
    empty_condition_rate = conf["empty_condition_rate"]
    num_labels = conf["num_labels"]
    channels = conf["channels"]
    width = conf["width"]
    # sampling conf
    saving_output_dir = conf["saving_output_dir"]
    inference_steps = conf["inference_steps"]
    guidance_scale = conf["guidance_scale"]
    num_samples_per_gpu = conf["num_samples_per_gpu"]
    op_epoch_interval = conf["op_epoch_interval"] 
    version = conf["version"]

    local_rank = rank

    torch.cuda.set_device(rank)

    setup(local_rank, world_size, "nccl")
    if (rank == 0):
        print("cluster inited")
    device = torch.device("cuda", local_rank)

    ddp_model = DDP(model_class(**model_conf).to(device), device_ids=[local_rank], output_device=local_rank)

    train_dataset = dataset_class(**dataset_conf)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=3, shuffle=(train_sampler is None))

    opt = torch.optim.AdamW(ddp_model.parameters(), lr=lr)
    slr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch)
    criterion = nn.MSELoss(reduction='mean')

    if(rank == 0):
        print("entry training loop")
    ddp_model.train()

    for e in range(epoch):
        # train
        train_sampler.set_epoch(e)
        for idx, (images, labels) in enumerate(train_dataloader):
            shape = images.shape[:-3] # (*, )
            z, y = images.to(device), labels.to(device) # (*, C, H, W) and (*,)
            # replace y with empty label with certain probability
            mask = torch.rand_like(y, device=y.device) < empty_condition_rate
            y[mask] = num_labels # num_labels = empty_label

            t = torch.rand(size=shape, device=device) # (*, )
            noise = torch.randn_like(z, device=device) # (*, C, H, W)
            
            alpha_t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (*, 1, 1, 1)
            beta_t = 1 - alpha_t # (*, 1, 1, 1)
            dalpha_t = torch.ones_like(alpha_t, device=device) * 1.0 # (*, 1, 1, 1)
            dbeta_t = torch.ones_like(alpha_t, device=device) * -1.0 # (*, 1, 1, 1)

            loss = criterion(ddp_model(alpha_t * z + beta_t * noise, t, y) , (dalpha_t * z + dbeta_t * noise))
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            # log
            dist.reduce(loss, 0)
            if (rank == 0):
                print(f"epoch {e}, step {idx}, avg_loss {loss / world_size}")
        # sample
        if (e % op_epoch_interval):
            samples = torch.randn(size=(num_samples_per_gpu, channels, width, width), device=device)

            y = torch.zeros(size=(num_samples_per_gpu,), device=device, dtype=torch.int64)
            empty_label = torch.zeros(size=(num_samples_per_gpu,), device=device, dtype=torch.int64) + num_labels

            samples = sampler(ddp_model, samples, inference_steps, guidance_scale, y, empty_label, True)

            prefix =  os.path.join(saving_output_dir, f"version{version}",f"epoch{e}")

            for i in range(samples.shape[0]):
                # output_dir/version/epoch/images.png
                save_tensor_to_image(samples, prefix, f"r{rank}_i{i}.png")
            if rank == 0:
                torch.save(ddp_model.module.state_dict(), os.path.join(prefix, f"checkpoint.pt"))
        slr.step()
        dist.barrier()
    cleanup()