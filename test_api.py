
import torch
import torch.nn as nn
import math

class dit(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # out_shape: (seq_len, d_model)
        self.register_buffer("pos_embd", self.get_sinusoidal_pos_emb(256, d_model))

 
    def forward(self, x): 
        print(self.pos_embd.shape)
        return x + self.pos_embd
        
    def get_sinusoidal_pos_emb(self, seq_len, d_model):
        position = torch.arange(0, seq_len).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe = torch.zeros(seq_len, d_model, requires_grad =False)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

d = dit(1024)
z = d(torch.zeros(size=(1, 256, 1024)))
print(z)