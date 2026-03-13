import torch
import torch.nn as nn
from einops import rearrange, einsum

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        # 目标：基于 theta 和 d_k，预先计算好尺寸为 (max_seq_len, d_k // 2) 的 cos 和 sin 矩阵
        # 
        # 1. 生成频段序号 k = [1, 2, ..., d_k/2]
        k = torch.arange(1,int(d_k/2)+1,device=device)
        # 2. 根据公式计算每个频段的 theta_k 频率系数
        theta_k = (2*k-2)/d_k
        # 3. 生成绝对位置序号 i = [0, 1, ..., max_seq_len - 1]
        i = torch.arange(max_seq_len,dtype=torch.float32,device=device)
        # 4. 计算所有位置和频段的角度组合矩阵 theta_{i,k}
        # i:seq theta_k:d_k/2 theta_ik:seq*(d_k/2)
        theta_ik = einsum(i,1/(theta**theta_k),"seq , d -> seq d")
        # 5. 求出该矩阵的 torch.cos() 和 torch.sin()
        cos = torch.cos(theta_ik)
        sin = torch.sin(theta_ik)
        
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播逻辑
        x 形状: (..., seq_len, d_k) - 注意这里可以处理任意数量的 batch 维度
        token_positions 形状: (..., seq_len)
        """
        
        # x: ... seq dk ,token_position : ... seq
        # sin_cached : seq dk/2
        # 1.  sin,cos:... seq d_k//2
        sin = self.sin_cached[token_positions]
        cos = self.cos_cached[token_positions]

        # 2. 维度对齐：提取出来的 cos/sin 形状可能需要通过 unsqueeze 增加一个
        #    特征维度，以便和 x 的形状 (..., seq_len, d_k) 进行正确的广播 (Broadcasting)。
        #    注意此时 cos/sin 的最后一维只有 d_k/2，你可能需要用 repeat_interleave 
        #    或者将 x 视为复数/变形为 2D 对来完成运算匹配。
        #   sin,cos:... seq_len d_k
        sin = sin.repeat_interleave(2,dim=-1)
        cos = cos.repeat_interleave(2,dim=-1)

        # 3. 旋转计算：执行类似 x1*cos - x2*sin, x1*sin + x2*cos 的两两配对旋转。
        #打包操作，每两个元素一打包，由... seq d_k变为... seq d_k/2 2
        x_paired = x.reshape(*x.shape[:-1],-1,2)
        #片取最后一维的向量，不保留最后一个维度
        x1 = x_paired[...,0]
        x2 = x_paired[...,1]
        #构造两两交换并一个取反的张量
        x3 = torch.stack([-x2,x1],dim=-1) #依然是... seq d_k/2 2
        x_swapped = x3.view_as(x) #用x的读法来读x_rotated,也就是看作... seq d_k
        #计算旋转后的Tensor
        x_rotated = x*cos + x_swapped*sin 
        
        return x_rotated