import numpy as np
import torch
from sklearn.datasets import make_s_curve

s_curve, _ = make_s_curve(10 ** 4, noise=0.1) # S型曲线数据集

# 将数据集中的特征缩放到一个相对较小的范围内，以便于模型的训练和收敛。
# 可以避免数据的特征值之间差异过大，导致某些特征对模型的影响过大，而其他特征的影响被忽略的情况。
# 同时，将数据的特征缩放到一个相对较小的范围内，也有助于提高模型的泛化能力，使其能够更好地适应新的未知数据。
s_curve = s_curve[:, [0, 2]] / 10.

print(F"shape of Moons:{np.shape(s_curve)}")

data = s_curve.T # (10000,2)->(2,10000)，每一列对应一个样本的所有特征值，更适合深度学习框架的输入格式，还可以保持数据的连续性
dataset = torch.Tensor(s_curve).float() # 保证输入数据类型的一致性，避免数据类型不匹配导致的错误。