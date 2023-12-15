import torch
from torch.optim import Adam
from models.networks import netD, netG

# setup optimizer
optimizerD = Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))