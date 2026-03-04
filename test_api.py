from train_dit import get_general_conf, get_model_conf, get_dataset_conf
from dit import dit
from trainer import ddp_trainer
import torch.multiprocessing as mp
from dataset import tensor_image_dataset

if __name__ == '__main__':
    conf = get_general_conf({})
    model_conf = get_model_conf(conf)
    dataset_conf = get_dataset_conf(conf)

    print("Entry training loop")
    mp.spawn(fn=ddp_trainer, args=(conf, dit, tensor_image_dataset, model_conf, dataset_conf), nprocs=conf["world_size"], join=True) # To be trained on 8*H800
    # create detailed log method
