import torch
import torch.nn as nn
from einops import rearrange, einsum


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones([d_model],device=device,dtype=dtype))#(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        ms = x.pow(2).mean(dim=-1, keepdim=True)#... , d_model->... , 1(dim=-1代表对最后一个维度求mean,keepdim表示保留1,否则就是...)
        rms = (ms+self.eps).sqrt()
        result = x/rms*self.weight#(... , d_model*d_model->...,d_model)
        return result.to(in_dtype)
