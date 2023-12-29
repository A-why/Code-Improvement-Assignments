import torch

num_step = 100  # 步数，一开始可以由beta、分布的均值和标准差来共同确定
batch_size = 512  # 批训练大小
num_epoch = 4000  # 定义迭代4000次

# 指定每一步的beta
betas = torch.linspace(-6, 6, num_step) # 生成等间隔的num_step个beta值
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5 # 转换为介于1e-5到0.5e-2之间的浮点数

# 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, dim=0) # 前t步的alpha值的累积乘积
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)  # p表示previous，前t-1步的alpha值的累积乘积
alphas_bar_sqrt = torch.sqrt(alphas_prod) # 前t步的alpha值的累积乘积的平方根
one_minus_alphas_bar_log = torch.log(1 - alphas_prod) # 前t步的alpha值的累积乘积的对数的负值
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod) # 前t步的alpha值的累积乘积的差值的平方根
