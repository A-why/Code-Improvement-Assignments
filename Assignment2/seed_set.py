import numpy as np
import torch
import random, numpy.random

# 设置随机种子, numpy, pytorch, python随机种子
def seed_torch(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True