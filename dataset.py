from torch.utils.data import Dataset
import os
import torch

class tensor_image_dataset(Dataset):
    def __init__(self, path_to_tokenzied_image_tensor, path_to_labels):
        assert os.path.exists(path_to_tokenzied_image_tensor) and os.path.exists(path_to_labels)
        self.data = torch.load(path_to_tokenzied_image_tensor, weights_only=True)
        if path_to_labels is not None:
            self.labels = torch.load(path_to_labels, weights_only=True).to(torch.int64) # int64 for embedding layer
        else:
            self.labels = torch.zeros(size=self.data.shape[:-3], dtype=torch.int64)

        # (B, C, H, W) and (B,)
        assert (len(self.data.shape) == 4) and (len(self.labels.shape) == 1)
        assert self.data.shape[0] == self.labels.shape[0]
        self.l = self.labels.shape[0]

    def __len__(self):
        return self.l
    
    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx]) # ((C, H, W), l)