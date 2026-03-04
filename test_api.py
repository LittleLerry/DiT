from train_dit import get_general_conf, get_model_conf, get_dataset_conf
from dit import dit
from trainer import ddp_trainer
import torch.multiprocessing as mp
from dataset import tensor_image_dataset
import torch

t = torch.load("./data/l.pt")
print(torch.max(t))
print(torch.min(t))
