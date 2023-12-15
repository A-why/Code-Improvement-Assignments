import torch

# rusume是否使用预训练模型继续训练,问号处输入模型的编号
resume = True  # 是继续训练，否重新训练

# GPU还是CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集名称
datasets = 'cifar10'

# 图片的通道数
nc = 3

# 类别数
n_classes = 10

# 控制生成器生成指定标签的图片
target_label = 4

# 训练批次数
batch_size = 128

# 噪声向量的维度
nz = 100

# 判别器的深度
ndf = 64
# 生成器的深度
ngf = 64

# 真实标签
real_label = 1.0
# 假标签
fake_label = 0.0
start_epoch = 0