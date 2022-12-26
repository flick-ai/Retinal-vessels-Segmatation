# 主控制文件，在此处输入网络控制命令
import torch
from torch import tensor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
from Dataset.dataset import MyDataset
import Args
from model.Mynet import BaselineUnet, Unet, AttentionUnet, SeUnet

# from monai.losses.dice import DiceLoss
from model.Loss import DiceLoss
from Function import read3D
from Template import Train
import torch
from axial_attention import AxialAttention


# 预设数据
batch_size = 1
learning_rate = 0.001
CUDA_on = True
cuda = CUDA_on and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
model = BaselineUnet(1, 1, 8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 导入数据
train_data3 = MyDataset(Args.Train_3D, loader=read3D, transform=ToTensor(), target_transform=ToTensor())
val_data3 = MyDataset(Args.Valid_3D, loader=read3D, transform=ToTensor(), target_transform=ToTensor(), valid=True)

train = DataLoader(train_data3, batch_size=batch_size, shuffle=True)
val = DataLoader(val_data3, batch_size=batch_size, shuffle=True)
net3D = Train(train, val, device, model, DiceLoss(), optimizer, 6, 5, "Net2D/Unet.pth")
net3D.train()
