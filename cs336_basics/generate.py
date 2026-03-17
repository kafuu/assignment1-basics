import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens, eos_id, temperature=1.0, top_p=1.0):
    """
    完整的文本生成逻辑，包含 Temperature 和 Top-p 采样。
    
    参数:
        model: 训练好的语言模型
        prompt_tokens: 起始 token，形状为 (batch_size, sequence_length)
        max_new_tokens: 允许生成的最大 token 数量
        eos_id: 结束符 <|endoftext|> 的 token ID
        temperature: 温度系数 (0.0 表示贪心搜索)
        top_p: 截断概率阈值
    """
    # 切换到评估模式，关闭 Dropout 等机制
    model.eval()
    x = prompt_tokens.clone()
    
    for _ in range(max_new_tokens):
        # 1. 前向传播获取 Logits
        # 假设 model(x) 返回形状为 (batch_size, sequence_length, vocab_size) 的张量
        logits = model(x)
        
        # 我们只需要序列最后一个时间步的输出作为预测依据
        next_token_logits = logits[:, -1, :] 
        
        # 2. Temperature 缩放
        if temperature == 0.0:
            # 绝对贪心搜索，直接拿最大值，跳过所有的采样逻辑
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            # 3. 转换为概率
            probs = F.softmax(next_token_logits, dim=-1)
            
            # 4. Top-p (Nucleus) 截断
            if top_p < 1.0:
                # 降序排序概率，并记录原始索引
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                # 计算累加概率和
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 创建需要被剔除的掩码 (Mask)：累加和超过 top_p 的长尾部分
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # 关键工程细节：把掩码向右平移一位。
                # 这样做是为了确保哪怕第一个词的概率就超过了 top_p，我们也至少保留这一个概率最大的词，防止整个池子被清空。
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 将被剔除的长尾部分概率强行赋为 0.0
                sorted_probs[sorted_indices_to_remove] = 0.0
                
                # 重新归一化，使得剩下的高优词概率加起来重新等于 1
                sorted_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)
                
                # 5. 在截断后干净的小池子里抽签 1 次
                next_token_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
                # 将排序后的抽签结果，映射回真实的原始 vocab ID
                next_token = torch.gather(sorted_indices, dim=-1, index=next_token_sorted_idx)
            else:
                # 如果不使用 Top-p，直接在原始概率分布上抽签
                next_token = torch.multinomial(probs, num_samples=1)
                
        # 6. 物理拼接与刹车检查
        x = torch.cat((x, next_token), dim=1)
        
        # 如果抽到了结束符，立刻打断自回归循环
        if next_token.item() == eos_id:
            break
            
    return x