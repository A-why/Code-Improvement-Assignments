import matplotlib.pyplot as plt
from datasets.datasets import *
from models.loss_function import diffusion_loss_fn
from models.parameters import *
from models.networks import MLPDiffusion
from models.parameters import alphas_bar_sqrt, one_minus_alphas_bar_sqrt
from trainers import trainer

# 计算任意时刻的x的采样值，基于x_0和参数重整化技巧
def q_x(x_0, t):
    # 可以基于x[0]得到任意时刻t的x[t]
    noise = torch.randn_like(x_0)  # noise是从正态分布中生成的随机噪声
    alphas_t = alphas_bar_sqrt[t]
    alphas_l_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_l_m_t * noise) # 在x[0]的基础上添加噪声

# 逆扩散采样函数
def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    # 从x[T]采样t时刻的重构值
    t = torch.tensor([t])
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt() # 标准差
    sample = mean + sigma_t * z
    return (sample)
def p_sample_loop(model, shape, n_step, betas, one_minus_alphas_bar_sqrt):
    # 从x[T]恢复x[T - 1]、x[T - 2]、...、x[0]
    # 从最后一个时刻T开始往前推，依次对每个时刻进行采样
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_step)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq

if __name__ == '__main__':
    # 检查所有变量的形状是否相同，并打印出 betas 变量的形状
    assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape == one_minus_alphas_bar_sqrt.shape
    print(f"all the same shape:{betas.shape}")
    seed = 2023  # 确保程序在每次运行时生成的随机数序列都是一样的

    # 参数平滑器,以便更好地泛化模型并减少过拟合
    # mu控制平滑程度，shadow是一个字典用于存储参数的平滑后的值
    class EMA():
        def __init__(self, mu=0.01):
            self.mu = mu
            self.shadow = {}

        # 将参数 val 注册到 shadow 字典中
        def register(self, name, val):
            self.shadow[name] = val.clone()

        # 对指定名称的参数 name 进行平滑处理
        def __call__(self, name, x):
            assert name in self.shadow
            new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
            return new_average

    print('Training model.....')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    plt.rc('text', color='blue')

    model = MLPDiffusion(num_step)  # 输出维度是2，输入是x和step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(num_epoch):
        for idx, batch_x in enumerate(dataloader):
            loss = trainer(model, batch_x, optimizer, diffusion_loss_fn)
        if (t % 100 == 0):
            print('epoch: %4d, loss: %.4f' % (t, loss.item()))
            x_seq = p_sample_loop(model, dataset.shape, num_step, betas, one_minus_alphas_bar_sqrt)  # 共有100个元素

            fig, axs = plt.subplots(1, 5, figsize=(28, 7))
            # 每20步画一张图
            for i in range(1, 6):
                cur_x = x_seq[i * 20].detach()
                axs[i - 1].scatter(cur_x[:, 0], cur_x[:, 1], color='red', edgecolor='white')
                axs[i - 1].set_axis_off()
                axs[i - 1].set_title('$q(\mathbf{x}_{' + str(i * 20) + '})$')
            plt.savefig('./figs/epoch_%d.png' % t)