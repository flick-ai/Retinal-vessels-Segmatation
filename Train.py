# 主控制文件，在此处输入网络控制命令
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
from construct.dataset import MyDataset
import Args
from model.Unet import U_Net

# 预设数据
batch_size = 2
learning_rate = 0.01
CUDA_on = True
cuda = CUDA_on and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
model = U_Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 导入数据
train_data = MyDataset(Args.Train, transform=ToTensor(), target_transform=ToTensor())
train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
model.train()


def train_func():
    for batch_idx, (data, target) in enumerate(train):
        data, target = data.float(), target.float()
        data, target = data.to(device), target.to(device)
        output = model(data)
        optimizer.zero_grad()
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch:[{}/{} ({:.0f}%)] \t Loss: {:.6f}'.
                  format(batch_idx * len(data), len(train_data),
                         100. * batch_idx / len(train_data), loss.item()))
        print(output)


train_func()
