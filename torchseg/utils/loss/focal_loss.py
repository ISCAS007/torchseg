import torch.nn.functional as F

def focal_loss(self, output, target, alpha, gamma, OHEM_percent):
    output = output.contiguous().view(-1)
    target = target.contiguous().view(-1)

    max_val = (-output).clamp(min=0)
    loss = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()

    # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
    invprobs = F.logsigmoid(-output * (target * 2 - 1))
    focal_loss = alpha * (invprobs * gamma).exp() * loss

    # Online Hard Example Mining: top x% losses (pixel-wise). Refer to https://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
    OHEM, _ = focal_loss.topk(k=int(OHEM_percent * [*focal_loss.shape][0]))
    return OHEM.mean()
