# 主控制文件，在此处输入网络控制命令
import torch
from torch import tensor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
from Dataset.dataset import MyDataset
import Args
from model.Mynet import BaselineUnet, Unet
from model.Loss import SoftDiceloss, DiceLoss, SoftIoULoss
from tqdm import tqdm
from Function import read3D
from Template import Train, Test

# 预设数据
batch_size = 1
CUDA_on = True
cuda = CUDA_on and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
model = torch.load("./Net3D/BasicUnet.pth")

# 导入数据
test_data = MyDataset(Args.Valid_3D, loader=read3D, transform=tensor, target_transform=ToTensor())
test = DataLoader(test_data, batch_size=batch_size, shuffle=True)
net3D = Test(test, device, model, "./Net3D/")
net3D.test()