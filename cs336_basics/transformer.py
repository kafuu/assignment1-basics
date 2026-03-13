import torch
import torch.nn as nn

from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.ffn import FeedForward
from cs336_basics.attention import CausalMultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 max_seq_len:int, 
                 theta: float,
                 device=None, 
                 dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model,device=device)
        self.attn = CausalMultiHeadAttention(d_model=d_model, 
                                             num_heads=num_heads, 
                                             max_seq_len=max_seq_len, 
                                             theta=theta)
        self.ln2 = RMSNorm(d_model,device=device)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff,device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播逻辑
        x: (batch, seq_len, d_model)
        """
        
        # Attention
        # 1. 归一化：
        h = self.ln1(x)
        # 2. 注意力提取：
        attn_out = self.attn(h, token_positions)
        # 3. 残差相加：
        x = x + attn_out 

        # FFN
        # 1. 归一化：
        h = self.ln2(x)
        # 2. 前馈计算：
        ffn_out = self.ffn(h)
        # 3. 残差相加：
        x = x + ffn_out

        return x