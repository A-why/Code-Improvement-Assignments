import torch
import torchvision.transforms as transforms
from models.parameters import device

apply_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 将标签进行one-hot编码
def to_categrical(y: torch.FloatTensor, lb):
    y_one_hot = lb.transform(y.cpu())
    floatTensor = torch.FloatTensor(y_one_hot)
    return floatTensor.to(device)
