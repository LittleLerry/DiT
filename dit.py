import torch
import torch.nn as nn

class lock():
    def __init__(self, path_to_lock):
        self.lock = path_to_lock

    def acquire(self,):
        pass

class attn(nn.Module):
    def __init__(self, d_model, num_heads, dp, ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dp , batch_first=True)
    def forward(self, x):
        out, _ = self.mha(x)
        return out

class mlp(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.dff = nn.Linear(d_model, d_ff)
        self.ffd = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
    def forward(self, x):
        x = self.gelu(self.dff(x))
        return self.ffd(x)

class ditblock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dp):
        super().__init__()
        #! elementwise_affine=False is importtant | why there are two ln?
        self.ln1 = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=False)
        #! not a mlp actually
        self.c_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model)
        )
        self.ffn = mlp(d_model, d_ff)
        self.mha = attn(d_model, num_heads, dp)

    def forward(self, x, c): # (*, seq_len, d_model), (*, d_model)
        # each (*, d_model)
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.c_mlp(c).chunk(6, dim=-1) # (*, 6 * d_model) 
        x = x + self.mha(self.ln1(x) * gamma1.unsqueeze(-2) + beta1.unsqueeze(-2)) * alpha1.unsqueeze(-2)
        x = x + self.ffn(self.ln2(x) * gamma2.unsqueeze(-2) + beta2.unsqueeze(-2)) * alpha2.unsqueeze(-2)
        return x

class dit(nn.Module):
    def __init__(self, num_blocks, d_in, d_model, d_ff, num_heads, h_in, w_in, patch_size, dp, num_labels):
        super().__init__()
        self.blocks = nn.Sequential(*[ditblock(d_model, d_ff, num_heads, dp) for _ in range(num_blocks)])
        assert (h_in % patch_size == 0) and (w_in % patch_size == 0)
        self.patch = patchify(d_in, patch_size, d_model)
        self.depatch = depatchify(d_model, patch_size, d_in, h_in, w_in)
        self.pos_embd = None # out_shape: (seq_len, d_model)
        self.t_embd = None # out_shape: (*, d_model), (*,)
        self.num_labels = num_labels
        self.y_embd = nn.Embedding(num_labels + 1, d_model) # out_shape: (*, d_model), (*,)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, t, y): 

        t_embded = self.t_embd(t) # type: ignore # (*, d_model)
        y_embded = self.y_embd(y) # type: ignore # (*, d_model)
        condition = t_embded + y_embded # (*, d_model)
        
        x = self.patch(x) + self.pos_embd # (*, c_in, h_in, w_in) -> (*, seq_len, d_model)
        for block in self.blocks:
            x = block(x, condition)
        #! no sigma
        x = self.ln(x)
        return self.depatch(x) # (*, seq_len, d_model) -> (*, c_in, h_in, w_in)
    
    def empty_label(self):
        return self.num_labels

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