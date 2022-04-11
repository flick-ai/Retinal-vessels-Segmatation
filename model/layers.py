import torch
from torch import nn
from torch.nn import functional as F


def projection(x):
    x_sum = torch.sum(x, dim=2)
    out = 255 * (x_sum - torch.min(x_sum)) / (torch.max(x_sum) - torch.min(x_sum))
    return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, bias=True, **kwargs)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x, inplace=True)
        return x


class Upconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Upconv2d, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, bias=True, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, bias=False, **kwargs)
        self.norm2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x, inplace=True)
        return x


class FRN(nn.Module):
    def __init__(self, ndim, num_features, eps=1e-6,
                 learnable_eps=False):
        super(FRN, self).__init__()
        shape = (1, num_features) + (1,) * (ndim - 2)
        print(shape)
        self.eps = nn.Parameter(torch.ones(*shape) * eps)
        print(eps)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.Tensor(*shape))
        self.beta = nn.Parameter(torch.Tensor(*shape))
        self.tau = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    def forward(self, x):
        print(x.dim())
        avg_dims = tuple(range(2, x.dim()))  # (2, 3)
        nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)


class FrnConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(FrnConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.norm1 = FRN(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, bias=False, **kwargs)
        self.norm2 = FRN(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x, inplace=True)
        return x

