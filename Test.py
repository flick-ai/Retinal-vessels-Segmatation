# 主控制文件，在此处可直接利用测试组测试网络性能
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
from Dataset.dataset import MyDataset
import Args
from Function import count_parameters,show3D
from tqdm import tqdm
import cv2


data = np.load(Args.Dataset+"/Data/3D/Train/10001.npy")
show3D(data)







# batch_size = 1
# train_data = MyDataset(Args.Train_2D, transform=ToTensor(), target_transform=ToTensor())
# train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# CUDA_on = True
# cuda = CUDA_on and torch.cuda.is_available()
# device = torch.device("cuda" if cuda else "cpu")
# model = torch.load('./Net2D/model.pth')
#
#
# def Write(out, name):
#     out = (out.cpu()).detach().numpy()
#     out = np.reshape(out, (400, 400, 1))
#     cv2.imwrite(name, out)
#
#
# print("Begin testing:")
# print("The computing device:", "GPU" if device.type == "cuda" else "CPU")
# print("Total number of parameters:{}".format(str(count_parameters(model))))
# for batch_idx, (data, target) in tqdm(enumerate(train), total=len(train)):
#     data, target = data.float(), target.float()
#     data, target = data.to(device), target.to(device)
#     output = model(data)
#     if batch_idx == 1:
#         Write(data, '1.png')
#         Write(target, '2.png')
#         Write(output, '3.png')
