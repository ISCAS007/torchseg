import torch.nn.functional as F
def focal_loss_ohem(input, target, alpha, gamma, OHEM_percent=0.1,ignore_index=255):
    b,c,h,w=input.shape
    
    target = target.contiguous().view(-1)
    input=input.permute(0,2,3,1).contiguous().view(-1, c)
    valid=(target!=ignore_index)
    
    input=input[valid.nonzero().squeeze()]
    target=target(valid)
    input = input.contiguous().view(-1)
    

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
    invprobs = F.logsigmoid(-input * (target * 2 - 1))
    focal_loss = alpha * (invprobs * gamma).exp() * loss

    # Online Hard Example Mining: top x% losses (pixel-wise). Refer to https://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
    OHEM, _ = focal_loss.topk(k=int(OHEM_percent * [*focal_loss.shape][0]))
    return OHEM.mean()
