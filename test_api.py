
import torch
import math
from dit import dit


data = torch.load("./data/i.pt")
print(torch.max(data))
print(torch.mean(data))
print(torch.std(data))