# -*- coding: utf-8 -*-
import torch

a = torch.rand(3, 4,3)
print(a.shape)
a_np = a.numpy()
print(a_np.shape)
a_cpu = a.data.cpu()
a_gpu = a.data.cuda()
print(a_cpu.shape)
print(a_gpu.shape)
print(a.cpu().numpy())
print(a.cuda())

b=a.max(1)[1]
print(b)