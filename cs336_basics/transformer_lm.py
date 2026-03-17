import torch
import math
import torch.nn as nn

from cs336_basics.embedding import Embedding
from cs336_basics.transformer import TransformerBlock,ModifiedBlock
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm


class TransformerLM(nn.Module):
    def __init__(self, 
                vocab_size: int,
                context_length: int,
                d_model: int,
                num_layers: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float,
                device=None):
        super().__init__()
        
        # 模块 1：输入嵌入 (Embeddings)
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model,device=device)
        
        # 模块 2：Transformer Blocks
        self.layers = nn.ModuleList([
            ModifiedBlock(d_model=d_model, 
                             num_heads=num_heads, 
                             d_ff=d_ff,
                             max_seq_len=context_length,
                             theta=rope_theta,
                             device=device) 
            for _ in range(num_layers)
         ])
        
        # 模块 3：输出层 (Output Head)
        self.ln_final = RMSNorm(d_model=d_model,device=device)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size,device=device)

    def _init_weights(self, module):
        """
        GPT-2 权重初始化法则
        """
        # 1. 基础线性层与嵌入层：均值 0，标准差 0.02
        if isinstance(module, Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
        # 2. 残差投影层降温缩放 (Residual Projection Scaling)
        # 遍历自身的所有参数，利用参数名称的后缀进行拦截
        # 注意：这里拦截的后缀涵盖了常见的 Attention 和 FFN 输出层命名习惯。
        # 如果你的 TransformerBlock 内部变量名不同（例如叫 output_proj），请在此处添加。
        for name, param in module.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("out_proj.weight") or name.endswith("c_proj.weight") or name.endswith("down_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=(0.02 / math.sqrt(2 * self.num_layers)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x 形状: (batch, seq_len) 里面装的是整型的 Token IDs
        """
        
        # 1. 查表：将整型 ID 转换为特征向量，并注入位置信息
        tensor = self.token_embeddings(x)
        
        # 2. 穿越：用一个 for 循环，让数据依次穿过所有的 Block
        for block in self.layers:
            tensor = block(tensor)
        
        # 3. 投射：归一化后，映射到词表空间
        tensor = self.ln_final(tensor)
        logits = self.lm_head(tensor)
        
        # 返回形状为 (batch, seq_len, vocab_size) 的张量
        return logits