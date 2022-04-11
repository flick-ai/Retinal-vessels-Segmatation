# 主控制文件，在此处输入网络控制命令
import torch
from torch import tensor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
from Dataset.dataset import MyDataset
import Args
from model.Mynet import BaselineUnet, Unet, FrnUnet
from model.Loss import SoftDiceloss, DiceLoss, SoftIoULoss
from Function import read3D
from Template import Train

# 预设数据
batch_size = 1
learning_rate = 0.001
CUDA_on = True
cuda = CUDA_on and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
model = FrnUnet(1, 1, 1).to(device)
# model = Unet(1, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 导入数据
train_data3 = MyDataset(Args.Train_3D, loader=read3D, transform=tensor, target_transform=ToTensor())
# train_data2 = MyDataset(Args.Train_2D, transform=ToTensor(), target_transform=ToTensor())
val_data3 = MyDataset(Args.Valid_3D, loader=read3D, transform=tensor, target_transform=ToTensor())
# val_data2 = MyDataset(Args.Valid_2D, transform=ToTensor(), target_transform=ToTensor())

train = DataLoader(train_data3, batch_size=batch_size, shuffle=True)
val = DataLoader(val_data3, batch_size=batch_size, shuffle=True)
net3D = Train(train, val, device, model, nn.MSELoss(), optimizer, 1, "./Net3D/")
net3D.train()
