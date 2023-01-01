# 新添加的文件
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
# import graphviz
# import netron
import nibabel as nib
from monai.metrics import compute_generalized_dice

# 预设数据
batch_size = 1
learning_rate = 0.001
CUDA_on = True
cuda = CUDA_on and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
model = BaselineUnet(1, 5, 8).to(device)    # Try to change it, the raw data is (1, 5, 8)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 导入数据
train_data3 = MyDataset(Args.Train_3D, loader=read3D, transform=tensor, target_transform=ToTensor())
val_data3 = MyDataset(Args.Valid_3D, loader=read3D, transform=tensor, target_transform=ToTensor(), valid=True)

train = DataLoader(train_data3, batch_size=batch_size, shuffle=True)
val = DataLoader(val_data3, batch_size=batch_size, shuffle=True)
net3D = torch.load("G:/term5/BI_proj/Proj/my-BraTS2020/NetSave/multi-Baseline.pth")
net3D.eval()

loss = []
with torch.no_grad():
    for batch_idx, (list_data, list_data2, list_data3, list_data4, list_label) in tqdm(enumerate(val), total=len(val)):
        net3D.eval()
        for data0,data1,data2,data3, target in zip(list_data, list_data2, list_data3, list_data4, list_label):
            data0,data1,data2,data3, target = data0.float(),data1.float(),data2.float(),data3.float(), target.float()
            data0,data1,data2,data3, target = data0.to(device),data1.to(device),data2.to(device),data3.to(device), target.to(device)
            output = net3D(data0, data1, data2, data3)
            target = target.permute(0, 2, 3, 1)
            output = output.permute(0, 2, 3, 1)

            loss.append(compute_generalized_dice(output.cpu(), target.cpu()))

print("The average loss is: ", sum(loss)/len(loss))

print("End to show the result")