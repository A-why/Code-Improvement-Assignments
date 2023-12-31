import torch
def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    """
    对任意时刻t进行采样计算loss
    param：
    model：模型
    x_0：初始状态
    alphas_bar_sqrt、one_minus_alphas_bar_sqrt： 参数
    n_steps：时间步数
    return：损失值
    """
    batch_size = x_0.shape[0]
    # 随机采样一个时刻t，为了提高训练效率，这里确保t不重复
    # 对一个batchsize样本生成随机的时刻t，覆盖到更多不同的t
    t = torch.randint(0, n_steps, size=(batch_size // 2,))
    t = torch.cat([t, n_steps - 1 - t], dim=0)  # [batch]
    t = t.unsqueeze(-1)  # [batch, 1]
    # x0的系数
    a = alphas_bar_sqrt[t]
    # eps的系数
    aml = one_minus_alphas_bar_sqrt[t]
    # 生成随机噪声eps
    e = torch.randn_like(x_0)
    # 构造模型的输入
    x = x_0 * a + e * aml
    # 送入模型，得到t时刻的随机噪声预测值
    output = model(x, t.squeeze(-1))
    # 与真实噪声一起计算误差，求平均值
    return (e - output).square().mean()