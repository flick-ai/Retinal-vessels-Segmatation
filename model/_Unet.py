# 网络模型文件，本文件添加我们需要的网络模型代码
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# ==========================Core Module================================
class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,k_size=3,stride=1,padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=k_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        x = F.elu(x)
        return x


# ==================================================================

