# 主控制文件，在此处输入网络控制命令
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
from Dataset.dataset import MyDataset
import Args
from model._Unet import U_Net
from model.Loss import SoftDiceloss
from tqdm import tqdm
from Function import count_parameters


# 预设数据
batch_size = 1
learning_rate = 0.0001
CUDA_on = True
cuda = CUDA_on and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
model = U_Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 导入数据
train_data = MyDataset(Args.Train_2D, transform=ToTensor(), target_transform=ToTensor())
train = DataLoader(train_data, batch_size=batch_size, shuffle=True)


def train_func():
    model.train()
    print("Begin training:")
    print("The computing device:", "GPU" if device.type == "cuda" else "CPU")
    print("Total number of parameters:{}".format(str(count_parameters(model))))
    for batch_idx, (data, target) in tqdm(enumerate(train), total=len(train)):
        data, target = data.float(), target.float()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = SoftDiceloss()(output, target)
        # print(loss)
        loss.backward()
        optimizer.step()


train_func()
torch.save(model, "./Net2D/model.pth")