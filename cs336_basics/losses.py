import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算交叉熵损失。
    logits 形状: batch, vocab_size
    targets 形状: (batch) 包含正确的词元索引 (整数)
    """
    
    # 1. 寻找最大值：沿着 vocab_size 维度 (dim=-1) 找到 logits 的最大值 (keepdim=True)。
    max_logits = torch.max(logits,dim = -1,keepdim=True).values
    # 2. 偏移平移：shifted_logits = logits - max_logits
    shifted_logits = logits - max_logits
    # 3. 计算对数指数和：计算 log( sum( exp(shifted_logits) ) )
    logsum = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1,keepdim=True)) #batch vocab -> batch
    
    # 1. 目标提取：利用 targets 张量里的索引，从原始的 logits 中
    x_true = torch.gather(logits,dim=-1,index=targets.unsqueeze(-1)) #batch vocab -> batch
    # 2. 最终公式：Loss_i = (LogSumExp 的结果) - (x_true 的得分)
    loss = logsum - x_true + max_logits
    # 3. 降维求和：对所有的 Loss_i 调用 torch.mean() 求出全局平均值。
    # return 计算出的平均交叉熵 (标量)
    return torch.mean(loss)