import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from seed_set import seed_torch
from models.parameters import resume, device, datasets, nc, n_classes, target_label, batch_size, nz, start_epoch
from models.networks import netD, netG, weights_init
from models.optimizers import optimizerD, optimizerG
from datasets.datasets import get_cifar_dataset
from datasets.preprocess import to_categrical
from trainers import trainer

if __name__ == '__main__':
    seed_torch()
    netD.apply(weights_init)
    netG.apply(weights_init)
    train_dataset = get_cifar_dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 定义损失函数
    criterion = torch.nn.BCELoss()

    # 显示16张图片
    image = torch.Tensor(train_dataset.data[:16] / 255).permute(0, 3, 1, 2)
    plt.imshow(torchvision.utils.make_grid(image, nrow=4).permute(1, 2, 0))

    lb = LabelBinarizer()
    lb.fit(list(range(0, n_classes)))

    # 如果继续训练，就加载预训练模型
    if resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('./checkpoint/GAN_%s_best.pth' % datasets)
        netG.load_state_dict(checkpoint['net_G'])
        netD.load_state_dict(checkpoint['net_D'])
        start_epoch = checkpoint['start_epoch']
    print('netG:', '\n', netG)
    print('netD:', '\n', netD)

    print('training on:   ', device, '   start_epoch', start_epoch)

    netD, netG = netD.to(device), netG.to(device)
    # 固定生成器，训练判别器
    for epoch in range(start_epoch, 500):
        for batch, (data, target) in enumerate(train_loader):
            loss_D1, loss_D2, lossG = trainer(data, target, netD, netG, optimizerD, optimizerG, criterion, device, lb)

            if batch % 10 == 0:
                print('epoch: %4d, batch: %4d, discriminator loss: %.4f, generator loss: %.4f'
                      % (epoch, batch, loss_D1.item() + loss_D2.item(), lossG.item()))

            # 每2个epoch保存图片
            if epoch % 2 == 0 and batch == 0:
                # 生成指定target_label的图片
                noise_z1 = torch.randn(data.size(0), nz, 1, 1).to(device)
                target3 = to_categrical(torch.full((data.size(0), 1), target_label), lb).unsqueeze(2).unsqueeze(
                    3).float()  # 加到噪声上
                noise_z = torch.cat((noise_z1, target3), dim=1)  # (N,nz+n_classes,1,1)

                fake_data = netG(noise_z.to(device))
                # 如果是单通道图片，那么就转成三通道进行保存
                if nc == 1:
                    fake_data = torch.cat((fake_data, fake_data, fake_data), dim=1)  # fake_data（N,1,H,W）->(N,3,H,W)
                # 保存图片
                data = fake_data.detach().cpu().permute(0, 2, 3, 1)
                data = np.array(data)

                # 保存单张图片，将数据还原
                data = (data * 0.5 + 0.5)

                plt.imsave('./generated_fake/%s/epoch_%d.png' % (datasets, epoch), data[0])
                torchvision.utils.save_image(fake_data[:16] * 0.5 + 0.5, './generated_fake/%s/epoch_%d_grid.png' % (datasets, epoch), nrow=4, normalize=True)
        # 保存模型
        state = {
            'net_G': netG.state_dict(),
            'net_D': netD.state_dict(),
            'start_epoch': epoch + 1
        }
        torch.save(state, './checkpoint/GAN_%s_best.pth' % (datasets))
        torch.save(state, './checkpoint/GAN_%s_best_copy.pth' % (datasets))

