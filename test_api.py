
import torch
import torch.nn as nn
import math

y = torch.randint(0,16,size=(2,32))
num_labels = 16
empty_condition_rate = 0.3
mask = torch.rand_like(y.float(), device=y.device) < empty_condition_rate
y[mask] = num_labels # num_labels = empty_label
print(y)