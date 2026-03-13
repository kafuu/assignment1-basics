import torch
import torch.nn as nn
from einops import rearrange, einsum

from jaxtyping import Bool, Float, Int
from torch import Tensor

class Linear(nn.Module):
    def __init__(self,
                  in_features:int, 
                  out_features:int, 
                  device:torch.device =None, 
                  dtype: torch.dtype=None):
        super().__init__()

        #为了防止梯度消失和梯度爆炸，方差=2/in+out
        std = (2/in_features + out_features) ** 0.5
        empty_weight = torch.empty([out_features,in_features],device=device,dtype=dtype)
        self.weight = nn.Parameter(empty_weight)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        return(einsum(x,self.weight,"... din, dout din -> ... dout"))
    
if __name__ == "__main__":
    l = Linear(10,20,torch.device("cpu"),torch.float32)
    print(l.weight)
    print("key",l.state_dict())