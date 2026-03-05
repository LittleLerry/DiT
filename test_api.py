
import torch
import torch.nn as nn
import math

@staticmethod
def timestep_embedding(t, d_model, max_period=10000, include_norm_factor=False):
    # 1. 计算频率基数（几何级数分布）
    half = d_model // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )  # 形状：(half,)
    
    # 2. 计算角度参数：t * freqs，支持任意 batch 形状
    args = t[..., None].float() * freqs  # 广播后形状：(*, half)
    
    # 3. 生成正弦和余弦嵌入并拼接
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # 形状：(*, 2*half)
    
    # 4. 如果 d_model 是奇数，补零
    if d_model % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
    
    # 5. 可选：应用归一化系数，使向量范数为 1（符合论文理论）
    if include_norm_factor:
        embedding = embedding * math.sqrt(2 / d_model)
    
    return embedding

print(timestep_embedding(torch.tensor([0.01,0.50], dtype=torch.int64), 1024))