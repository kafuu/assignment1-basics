import torch
import torch.nn as nn
from einops import rearrange, einsum
from math import sqrt

from cs336_basics.rope import RoPE


def softmax(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    对输入张量的指定维度执行Softmax。
    """
    
    # 1. 寻找维度最大值：使用 torch.max 沿着参数传入的 `dim` 维度找到最大值。
    max_tensor = torch.amax(input=tensor,dim=dim,keepdim=True)
    # 2. 偏移平移：将原始 tensor 减去刚刚求出的最大值张量。
    tensor = tensor - max_tensor
    # 3. 求指数：对相减后的张量应用 torch.exp()。
    tensor_exp = torch.exp(tensor)
    # 4. 求和：使用 torch.sum 沿着 `dim` 维度对指数张量求和。
    tensor_sum = tensor_exp.sum(dim=dim,keepdim=True)
    # 5. 归一化：将第 3 步算出的指数张量，除以第 4 步算出的求和张量。
    tensor_softmax = tensor_exp/tensor_sum
    return tensor_softmax

    
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    """
    纯粹的 Attention 核心公式计算
    注意：这里传入的 q, k, v 已经是拆分好多头、且经过 RoPE 旋转后的纯净状态
    """
    # 伪代码逻辑：
    # 1. 执行矩阵乘法 q @ k.transpose(...)
    mul = einsum(q,k,"... queries d_k , ... keys d_k -> ... queries keys")
    # 2. 除以缩放因子 sqrt(d_k)
    mul = mul/sqrt(q.size(dim=-1))
    # 3. 【Mask 注入点】：如果传入了 mask，使用 masked_fill_() 将不允许看到的地方替换为 -inf
    if mask != None:
        mul = mul.masked_fill_(mask==False,value=-torch.inf)
    # 4. 【Softmax 发生地】：在这里对最后一步的结果调用 F.softmax()
    mul = softmax(mul,dim=-1)
    # 5. 将 Softmax 的概率矩阵去乘以 v
    return einsum(mul,v,"... a b, ... b c ->... a c")
    


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len:int=None, theta:float=None):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.q_proj = nn.Linear(d_model,d_model,bias=False)
        self.k_proj = nn.Linear(d_model,d_model,bias=False)
        self.v_proj = nn.Linear(d_model,d_model,bias=False)
        self.output_proj = nn.Linear(d_model,d_model,bias=False)
        if theta != None:
            self.rope = RoPE(theta=theta,d_k=d_model/num_heads,max_seq_len=max_seq_len) # 实例化你刚才写的 RoPE 类
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None):
        #多头注意力写到一个矩阵里
        # 1. 将 x 穿过 q_proj, k_proj, v_proj，得到原始的大 Q, K, V
        Q = einsum(x,self.q_proj.weight,"... seq d_model , d_k d_model -> ... seq d_k")
        K = einsum(x,self.k_proj.weight,"... seq d_model , d_k d_model -> ... seq d_k")
        V = einsum(x,self.v_proj.weight,"... seq d_model , d_v d_model -> ... seq d_v")

        # 2. 把 d_model 拆分成 num_heads 和 d_k，进行一系列的 view 和 transpose，
        #    把多头维度单独剥离出来，准备并行计算。
        d = int(Q.size(-1)/self.num_heads)
        Q = rearrange(Q,"... seq (num len) -> ... num seq len",len=d,num=self.num_heads)#... seq (num_heads m/h)
        K = rearrange(K,"... seq (num len) -> ... num seq len",len=d,num=self.num_heads)
        V = rearrange(V,"... seq (num len) -> ... num seq len",len=d,num=self.num_heads)

        if token_positions is not None and self.rope is not None:
            Q = self.rope(Q,token_positions)#不需要.forward，因为继承了nn.module类，__call__方法会引到自己写的forward
            K = self.rope(K,token_positions)
        else: 
            if token_positions is None and self.rope is not None:
                Q = self.rope(Q,torch.arange(Q.size(-2),device=Q.device))
                K = self.rope(K,torch.arange(K.size(-2),device=K.device))


        # 4. 生成因果掩码：构建一个右上角全是 -inf 的下三角矩阵 (Causal Mask)
        mask = torch.tril(torch.ones(Q.size(-2),Q.size(-2),device=x.device,dtype=torch.bool))

        out = scaled_dot_product_attention(Q, K, V, mask)

        out = rearrange(out,"... b a c -> ... a (b c)")

        return einsum(out,self.output_proj.weight,"... seq d_v , d_model d_v -> ... seq d_model")
