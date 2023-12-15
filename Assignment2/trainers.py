import torch
from models.parameters import nz, real_label, fake_label
from datasets.preprocess import to_categrical

def trainer(data, target, netD, netG, optimizerD, optimizerG, criterion, device, lb):
    data = data.to(device)
    target = target.to(device)
    # 拼接真实数据和标签
    target1 = to_categrical(target, lb).unsqueeze(2).unsqueeze(3).float()  # 加到噪声上
    target2 = target1.repeat(1, 1, data.size(2), data.size(3))  # 加到数据上
    data = torch.cat((data, target2),
                     dim=1)  # 将标签与数据拼接 (N,nc,128,128),(N,n_classes, 128,128)->(N,nc+nc_classes,128,128)

    label = torch.full((data.size(0), 1), real_label).to(device)

    # 1、训练判别器
    # training real data
    netD.zero_grad()
    output = netD(data)
    loss_D1 = criterion(output, label)
    loss_D1.backward()

    # training fake data,拼接噪声和标签
    noise_z = torch.randn(data.size(0), nz, 1, 1).to(device)
    noise_z = torch.cat((noise_z, target1), dim=1)  # (N,nz+n_classes,1,1)
    # 拼接假数据和标签
    fake_data = netG(noise_z)
    fake_data = torch.cat((fake_data, target2), dim=1)  # (N,nc+n_classes,128,128)
    label = torch.full((data.size(0), 1), fake_label).to(device)

    output = netD(fake_data.detach())
    loss_D2 = criterion(output, label)
    loss_D2.backward()

    # 更新判别器
    optimizerD.step()

    # 2、训练生成器
    netG.zero_grad()
    label = torch.full((data.size(0), 1), real_label).to(device)
    output = netD(fake_data.to(device))
    lossG = criterion(output, label)
    lossG.backward()

    # 更新生成器
    optimizerG.step()

    return loss_D1, loss_D2, lossG