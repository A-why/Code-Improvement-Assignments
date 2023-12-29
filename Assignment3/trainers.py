import torch.nn as nn
from models.parameters import *

def trainer(model, batch_x, optimizer, loss_fn):
    loss = loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_step)
    optimizer.zero_grad()  # 对梯度进行清零，防止网络权重更新过于迅速或不稳定，无法得到正确的收敛结果
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.)  # 对梯度进行裁剪，避免出现梯度爆炸
    optimizer.step()
    return loss