import torch
from torch import nn
from torch.nn import functional as F
import cv2
from axial_attention import AxialAttention, SelfAttention


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
        self.eps = nn.Parameter(torch.ones(*shape) * eps)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.Tensor(*shape))
        self.beta = nn.Parameter(torch.Tensor(*shape))
        self.tau = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    def forward(self, x):
        print(x.dim())
        avg_dims = tuple(range(2, x.dim()))
        nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        print(nu2 + torch.abs(self.eps))
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)


class AttConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(AttConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, bias=False, **kwargs)
        self.norm2 = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x, inplace=True)
        return x

class AttConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(AttConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, bias=False, **kwargs)
        self.norm2 = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x, inplace=True)
        return x

class FastSmoothSENorm(nn.Module):
    class SEWeights(nn.Module):
        def __init__(self, in_channels, reduction=2):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        def forward(self, x):
            b, c, h, w = x.size()
            out = torch.mean(x.view(b, c, -1), dim=-1).view(b, c, 1, 1)  # output_shape: in_channels x (1, 1, 1)
            out = F.relu(self.conv1(out))
            out = self.conv2(out)
            return out

    def __init__(self, in_channels, reduction=2):
        super(FastSmoothSENorm, self).__init__()
        self.norm = nn.InstanceNorm2d(in_channels, affine=False)
        self.gamma = self.SEWeights(in_channels, reduction)
        self.beta = self.SEWeights(in_channels, reduction)

    def forward(self, x):
        gamma = torch.sigmoid(self.gamma(x))
        beta = torch.tanh(self.beta(x))
        x = self.norm(x)
        return gamma * x + beta


class FastSmoothSeNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super(FastSmoothSeNormConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.norm = FastSmoothSENorm(out_channels, reduction)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.norm(x)
        return x


class FastSmoothSeNormConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super(FastSmoothSeNormConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=True, **kwargs)
        self.norm = FastSmoothSENorm(out_channels, reduction)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.norm(x)
        return x


class RESseNormConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super().__init__()
        self.conv1 = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, **kwargs)

        if in_channels != out_channels:
            self.res_conv = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, kernel_size=1, stride=1,
                                                   padding=0)
        else:
            self.res_conv = None

    def forward(self, x):
        residual = self.res_conv(x) if self.res_conv else x
        x = self.conv1(x)
        x += residual
        return x

class RESseNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super().__init__()
        self.conv1 = FastSmoothSeNormConv2d(in_channels, out_channels, reduction, **kwargs)

        if in_channels != out_channels:
            self.res_conv = FastSmoothSeNormConv2d(in_channels, out_channels, reduction, kernel_size=1, stride=1,
                                                   padding=0)
        else:
            self.res_conv = None

    def forward(self, x):
        residual = self.res_conv(x) if self.res_conv else x
        x = self.conv1(x)
        x += residual
        return x

class SEConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(SEConv3d, self).__init__()
        self.conv1 = RESseNormConv3d(in_channels, out_channels, reduction=2, **kwargs)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.conv2 = RESseNormConv3d(out_channels, out_channels, reduction=2, **kwargs)
        self.norm2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x, inplace=True)
        return x

class SEConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(SEConv2d, self).__init__()
        self.conv1 = RESseNormConv2d(in_channels, out_channels, reduction=2, **kwargs)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = RESseNormConv2d(out_channels, out_channels, reduction=2, **kwargs)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x, inplace=True)
        return x


# The original code

# class BasicConv3d(nn.Module):
#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(BasicConv3d, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
#         self.norm1 = nn.BatchNorm3d(out_channels)
#         self.conv2 = nn.Conv3d(out_channels, out_channels, bias=False, **kwargs)
#         self.norm2 = nn.BatchNorm3d(out_channels)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = F.relu(x, inplace=True)
#         x = self.conv2(x)
#         x = self.norm2(x)
#         x = F.relu(x, inplace=True)
#         return x


# class FRN(nn.Module):
#     def __init__(self, ndim, num_features, eps=1e-6,
#                  learnable_eps=False):
#         super(FRN, self).__init__()
#         shape = (1, num_features) + (1,) * (ndim - 2)
#         self.eps = nn.Parameter(torch.ones(*shape) * eps)
#         if not learnable_eps:
#             self.eps.requires_grad_(False)
#         self.gamma = nn.Parameter(torch.Tensor(*shape))
#         self.beta = nn.Parameter(torch.Tensor(*shape))
#         self.tau = nn.Parameter(torch.Tensor(*shape))
#         self.reset_parameters()

#     def forward(self, x):
#         print(x.dim())
#         avg_dims = tuple(range(2, x.dim()))
#         nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
#         print(nu2 + torch.abs(self.eps))
#         x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
#         return torch.max(self.gamma * x + self.beta, self.tau)

#     def reset_parameters(self):
#         nn.init.ones_(self.gamma)
#         nn.init.zeros_(self.beta)
#         nn.init.zeros_(self.tau)


# class AttConv3d(nn.Module):
#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(AttConv3d, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
#         self.norm1 = nn.GroupNorm(1, out_channels)
#         self.conv2 = nn.Conv3d(out_channels, out_channels, bias=False, **kwargs)
#         self.norm2 = nn.GroupNorm(1, out_channels)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = F.relu(x, inplace=True)
#         x = self.conv2(x)
#         x = self.norm2(x)
#         x = F.relu(x, inplace=True)
#         return x


# class FastSmoothSENorm(nn.Module):
#     class SEWeights(nn.Module):
#         def __init__(self, in_channels, reduction=2):
#             super().__init__()
#             self.conv1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=True)
#             self.conv2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

#         def forward(self, x):
#             b, c, d, h, w = x.size()
#             out = torch.mean(x.view(b, c, -1), dim=-1).view(b, c, 1, 1, 1)  # output_shape: in_channels x (1, 1, 1)
#             out = F.relu(self.conv1(out))
#             out = self.conv2(out)
#             return out

#     def __init__(self, in_channels, reduction=2):
#         super(FastSmoothSENorm, self).__init__()
#         self.norm = nn.InstanceNorm3d(in_channels, affine=False)
#         self.gamma = self.SEWeights(in_channels, reduction)
#         self.beta = self.SEWeights(in_channels, reduction)

#     def forward(self, x):
#         gamma = torch.sigmoid(self.gamma(x))
#         beta = torch.tanh(self.beta(x))
#         x = self.norm(x)
#         return gamma * x + beta


# class FastSmoothSeNormConv3d(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
#         super(FastSmoothSeNormConv3d, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, bias=True, **kwargs)
#         self.norm = FastSmoothSENorm(out_channels, reduction)

#     def forward(self, x):
#         x = self.conv(x)
#         x = F.relu(x, inplace=True)
#         x = self.norm(x)
#         return x


# class RESseNormConv3d(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
#         super().__init__()
#         self.conv1 = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, **kwargs)

#         if in_channels != out_channels:
#             self.res_conv = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, kernel_size=1, stride=1,
#                                                    padding=0)
#         else:
#             self.res_conv = None

#     def forward(self, x):
#         residual = self.res_conv(x) if self.res_conv else x
#         x = self.conv1(x)
#         x += residual
#         return x


# # class UpConv(nn.Module):
# #     def __init__(self, in_channels, out_channels, reduction=2, scale=2):
# #         super().__init__()
# #         self.scale = scale
# #         self.conv = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, kernel_size=1, stride=1, padding=0)
# #
# #     def forward(self, x):
# #         x = self.conv(x)
# #         x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
# #         return x


# class SEConv3d(nn.Module):
#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(SEConv3d, self).__init__()
#         self.conv1 = RESseNormConv3d(in_channels, out_channels, reduction=2, **kwargs)
#         self.norm1 = nn.BatchNorm3d(out_channels)
#         self.conv2 = RESseNormConv3d(out_channels, out_channels, reduction=2, **kwargs)
#         self.norm2 = nn.BatchNorm3d(out_channels)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = F.relu(x, inplace=True)
#         x = self.conv2(x)
#         x = self.norm2(x)
#         x = F.relu(x, inplace=True)
#         return x


# class UpConv(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction=2, scale=2):
#         super().__init__()
#         self.scale = scale
#         self.conv = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
#         return x
