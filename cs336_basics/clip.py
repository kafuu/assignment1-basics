def clip_gradients(parameters, max_norm: float):
    valid_params = [p for p in parameters if p.grad is not None]
    
    # 遍历 valid_params 中的每一个 p
    total_norm = 0
    for p in valid_params:
        grad = p.grad
        # 取出 p.grad，计算它的平方和
        total_norm += grad.pow(2).sum()

    
    if total_norm > max_norm:
        clip =  max_norm / total_norm 

    for p in valid_params:
        p.grad.mul_(clip)
        