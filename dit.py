import torch
import torch.nn as nn
import math
import os
# TODO EMA
# does it critical
def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class attn(nn.Module):
    def __init__(self, d_model, num_heads, dp, ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dp , batch_first=True)
    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out

class mlp(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.dff = nn.Linear(d_model, d_ff)
        self.ffd = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU(approximate="tanh") # from the paper
    def forward(self, x):
        x = self.gelu(self.dff(x))
        return self.ffd(x)

class ditblock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dp):
        super().__init__()
        #! elementwise_affine=False is importtant | why there are two ln?
        self.ln1 = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=False)

        # same in the paper
        self.c_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model) # zero init
        )

        nn.init.constant_(self.c_mlp[1].weight, 0) # type: ignore
        nn.init.constant_(self.c_mlp[1].bias, 0) # type: ignore

        self.ffn = mlp(d_model, d_ff)
        self.mha = attn(d_model, num_heads, dp)

    def forward(self, x, c): # (*, seq_len, d_model), (*, d_model)
        # each (*, d_model)
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.c_mlp(c).chunk(6, dim=-1) # (*, 6 * d_model) 
        x = x + self.mha(modulate(self.ln1(x), beta1.unsqueeze(-2), gamma1.unsqueeze(-2))) * alpha1.unsqueeze(-2) # critical!
        x = x + self.ffn(modulate(self.ln2(x), beta2.unsqueeze(-2), gamma2.unsqueeze(-2))) * alpha2.unsqueeze(-2)
        return x

class dit(nn.Module):
    def __init__(self, num_blocks, d_in, d_model, d_ff, num_heads, h_in, w_in, patch_size, dp, num_labels):
        super().__init__()
        assert (h_in % patch_size == 0) and (w_in % patch_size == 0)
        assert d_model % 2 == 0

        self.blocks = nn.Sequential(*[ditblock(d_model, d_ff, num_heads, dp) for _ in range(num_blocks)])
        num_patches = (h_in // patch_size) * (w_in // patch_size)
        self.patch = patchify(d_in, patch_size, d_model)
        self.depatch = depatchify(d_model, patch_size, d_in, h_in, w_in)

        # out_shape: (seq_len, d_model)
        self.register_buffer("pos_embd", self.get_sinusoidal_pos_emb(num_patches, d_model))

        self.t_embd = time_embd(d_model) # out_shape: (*,) -> (*, d_model)
        self.num_labels = num_labels
        self.y_embd = nn.Embedding(num_labels + 1, d_model) # out_shape: (*, d_model), (*,)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, t, y): 

        t_embded = self.t_embd(t) # (*, d_model)
        y_embded = self.y_embd(y) # (*, d_model)
        condition = t_embded + y_embded # (*, d_model)
        
        x = self.patch(x) + self.pos_embd # (*, c_in, h_in, w_in) -> (*, seq_len, d_model)
        for block in self.blocks:
            x = block(x, condition)
        #! no sigma
        x = self.ln(x)
        return self.depatch(x) # (*, seq_len, d_model) -> (*, c_in, h_in, w_in)
    
    def empty_label(self):
        return self.num_labels
    
    def get_sinusoidal_pos_emb(self, seq_len, d_model):
        position = torch.arange(0, seq_len).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe = torch.zeros(seq_len, d_model, requires_grad =False)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

class patchify(nn.Module):
    def __init__(self, in_channels, patch_size, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=patch_size,
                               stride=patch_size,)
    # (*, C, H, W) -> (*, seq_len, d_model)
    def forward(self, x):
        return self.conv1(x).flatten(-2).transpose(-1,-2)

"""
def unpatchify(self, x):
    c = self.out_channels
    p = self.x_embedder.patch_size[0]
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs
"""

# TODO
class depatchify(nn.Module):
    def __init__(self, in_channels, patch_size, out_channels, h, w):
        super().__init__()
        self.num_patches_h = h // patch_size
        self.num_patches_w = w // patch_size
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels, 
                                        out_channels=out_channels, 
                                        kernel_size=patch_size, 
                                        stride=patch_size)
    
    # (*, seq_len, d_model) -> (*, C, H, W)
    def forward(self, x):
        shape = x.shape[:-2]
        x = x.transpose(-1,-2).view(*shape, -1, self.num_patches_h, self.num_patches_w)
        return self.conv1(x)

# use paper implmentation
class time_embd(nn.Module):
    def __init__(self, d_model, d_t=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_t, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.d_t = d_t
    # TODO FIX BUG
    @staticmethod
    def freq_embd(t, dim, max_period=10000): # (*,)
        t = t * max_period # FUCK YOU, embedding can be bitch

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t):
        t_freq = self.freq_embd(t, self.d_t)
        return self.mlp(t_freq)