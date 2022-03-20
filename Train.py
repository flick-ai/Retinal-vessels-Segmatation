# 主控制文件，在此处输入网络控制命令
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
from Dataset.dataset import MyDataset
import Args
from model.Unet import U_Net
from model.Loss import SoftDiceloss, DiceLoss, CrossEntropyLoss2d
from tqdm import tqdm
from Function import count_parameters
from Class import Train

# 预设数据
batch_size = 1
learning_rate = 0.01
CUDA_on = True
cuda = CUDA_on and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
model = U_Net(1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 导入数据
train_data = MyDataset(Args.Train_2D, transform=ToTensor(), target_transform=ToTensor())
train = DataLoader(train_data, batch_size=batch_size, shuffle=True)

net2D = Train(train, device, model, nn.MSELoss(), optimizer, 20, "./Net2D/model.pth")
net2D.train()