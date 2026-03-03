from dit import dit
from trainer import ddp_trainer
import torch.multiprocessing as mp
from dataset import tensor_image_dataset

def get_general_conf():
    conf = {
        "world_size": 8,
        "batch_size": 32,
        "lr": 1e-4,
        "epoch": 512,
        "empty_condition_rate": 0.1,
        "num_labels": 1000,
        "channels": 4,
        "width": 32,
        "saving_output_dir": "./output",
        "inference_steps": 1000,
        "guidance_scale": 4.0,
        "num_samples_per_gpu": 4,
        "op_epoch_interval": 32,
        "version": 1,
    }
    return conf

def get_model_conf():
    # def __init__(self, num_blocks, d_in, d_model, d_ff, num_heads, h_in, w_in, patch_size, dp, num_labels):
    conf = {
        "num_blocks": 28,
        "d_in": 4,
        "d_model": 1024,
        "d_ff": 2048,
        "num_heads": 16,
        "h_in": 32,
        "w_in": 32,
        "patch_size": 2,
        "dp": 0.0,
        "num_labels": 1000,
    }
    return conf

def get_dataset_conf():
    conf = {
        "path_to_tokenzied_image_tensor": "./data/i.pt",
        "path_to_labels": "./data/l.pt",
    }
    return conf

if __name__ == '__main__':
    num_processes = 8
    conf = get_general_conf()
    model_conf = get_model_conf()
    dataset_conf = get_dataset_conf()

    print("Entry training loop")
    mp.spawn(fn=ddp_trainer, args=(conf, dit, tensor_image_dataset, model_conf, dataset_conf), nprocs=num_processes, join=True) # To be trained on 8*H800