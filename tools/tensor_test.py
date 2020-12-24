# -*- coding: utf-8 -*-
import torch
import numpy as np

a = torch.rand(3,4,3)
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

img_a=np.random.randint(20,size=(3,10,10))
img_b=np.random.randint(20,size=(3,10,10))
img_c=(img_a==img_b).astype(np.float)

torch_img_c=torch.from_numpy(img_c)
print(torch_img_c)