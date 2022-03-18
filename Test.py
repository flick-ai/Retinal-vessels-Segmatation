# 主控制文件，在此处可直接利用测试组测试网络性能
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
from Dataset.dataset import MyDataset
import Args
from Function import count_parameters
from tqdm import tqdm
import cv2

batch_size = 1
train_data = MyDataset(Args.Train_2D, transform=ToTensor(), target_transform=ToTensor())
train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
CUDA_on = True
cuda = CUDA_on and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
model = torch.load('./Net2D/model.pth')

print("Begin testing:")
print("The computing device:", "GPU" if device.type == "cuda" else "CPU")
print("Total number of parameters:{}".format(str(count_parameters(model))))
for batch_idx, (data, target) in tqdm(enumerate(train), total=len(train)):
    data, target = data.float(), target.float()
    data, target = data.to(device), target.to(device)
    output = model(data)
    if batch_idx == 1:
        print((output.cpu()).detach().numpy())
        # cv2.imwrite('1.png', ()
