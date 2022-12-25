# 主控制文件，在此处输入网络控制命令
import torch
from torch import tensor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
from Dataset.dataset import MyDataset
import Args
from model.Mynet import BaselineUnet, Unet
from model.Loss import DiceLoss
from tqdm import tqdm
from Function import read3D, show3D
import torchvision
from Template import Train, Test
import graphviz
import netron
import nibabel as nib

# 预设数据
# batch_size = 1
# CUDA_on = True
# cuda = CUDA_on and torch.cuda.is_available()
# device = torch.device("cuda" if cuda else "cpu")
# model = torch.load("./Net2D/Unet.pth")
#
# # 导入数据
# test_data = MyDataset(Args.Train_3D, loader=read3D, transform=tensor, target_transform=ToTensor())
# test = DataLoader(test_data, batch_size=batch_size, shuffle=False)
# net3D = Test(test, device, model, "./Net2D/")
# net3D.test()
path = "BraTS20_Training_001.nii.gz"
img = nib.load(path).get_fdata()
show3D(img)
