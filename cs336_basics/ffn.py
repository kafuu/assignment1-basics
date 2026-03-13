import torch
import torch.nn as nn
from einops import rearrange, einsum

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int=None, device=None, dtype=None):
        super().__init__()
        if d_ff == None:
            d_ff = int(8 * d_model / 3)
        """
        FFN(x) = W2( SiLU(W1(x)) ⊙ W3(x) )
        """
        self.w1 = nn.Linear(in_features=d_model, out_features=d_ff, bias=False,device=device,dtype=dtype)#升维矩阵(d_model->d_ff)
        self.w2 = nn.Linear(in_features=d_ff, out_features=d_model, bias=False,device=device,dtype=dtype)#降维矩阵(d_ff->d_model)
        self.w3 = nn.Linear(in_features=d_model, out_features=d_ff, bias=False,device=device,dtype=dtype)#门控GLU矩阵(d_model->d_ff)
        pass

    def forward(self, x):
        """
        FFN(x) = W2( SiLU(W1(x)) ⊙ W3(x) )
        """
        #x : ... , d_model
        # 1. 将 x 分别送入 w1 和 w3 进行升维
        w1x = einsum(x,self.w1.weight,"... in , out in -> ... out")
        w3x = einsum(x,self.w3.weight,"... in , out in -> ... out")
        # 2. 对 w1 的输出结果应用 F.silu 激活函数
        silued_w1 = w1x * torch.sigmoid(w1x)
        # 3. 将激活后的 w1 结果与 w3 的结果进行逐元素相乘 (*)
        mul = silued_w1 * w3x
        # 4. 将相乘的结果送入 w2 进行降维，作为最终结果返回
        res = einsum(mul,self.w2.weight,"... in , out in -> ... out")
        return res