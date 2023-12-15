import torch, torchvision
from datasets.preprocess import apply_transform
from torchvision.datasets import CIFAR10

def get_cifar_dataset():
    train_dataset = CIFAR10(root='../data/cifar10', train=False, download=True, transform=apply_transform)
    return train_dataset