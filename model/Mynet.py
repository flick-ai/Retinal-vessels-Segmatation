import torch
from torch import nn
from torch.nn import functional as F
from .layers import BasicConv3d, BasicConv2d, Upconv2d, projection, FrnConv3d
import cv2
from Function import show3D


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters=64):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.out_channels = out_channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = BasicConv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.Conv2 = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Conv3 = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Conv4 = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Conv5 = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        self.Up4 = Upconv2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.decoder4 = BasicConv2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Up3 = Upconv2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.decoder3 = BasicConv2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Up2 = Upconv2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.decoder2 = BasicConv2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Up1 = Upconv2d(2 * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.decoder1 = BasicConv2d(2 * n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv2d(n_filters, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if len(x.shape) == 5:
            x = projection(x)
        out = (x.cpu()).detach().numpy()
        out = out[0, 0, :, :]
        cv2.imwrite('x.png', out)
        e1 = self.Conv1(x)
        e2 = self.Conv2(self.pool(e1))
        e3 = self.Conv3(self.pool(e2))
        e4 = self.Conv4(self.pool(e3))
        x = self.Conv5(self.pool(e4))
        x = self.decoder4(torch.cat((self.Up4(x), e4), 1))
        x = self.decoder3(torch.cat((self.Up3(x), e3), 1))
        x = self.decoder2(torch.cat((self.Up2(x), e2), 1))
        x = self.decoder1(torch.cat((self.Up1(x), e1), 1))
        x = self.conv1x1(x)
        return x


class BaselineUnet(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters):
        super(BaselineUnet, self).__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.out_channels = out_channels

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder1 = BasicConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.encoder2 = BasicConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.encoder3 = BasicConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.encoder4 = BasicConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.Up3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.decoder3 = BasicConv3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Up2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.decoder2 = BasicConv3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Up1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = BasicConv3d(2 * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv3d(n_filters, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, test=False):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        x = self.encoder4(self.pool(e3))
        x = self.decoder3(torch.cat([self.Up3(x), e3], 1))
        x = self.decoder2(torch.cat([self.Up2(x), e2], 1))
        x = self.decoder1(torch.cat([self.Up1(x), e1], 1))
        x = self.conv1x1(x)
        if test:
            return x
        x = projection(x)
        return x


class FrnUnet(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters):
        super(FrnUnet, self).__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.out_channels = out_channels

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder1 = FrnConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.encoder2 = FrnConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.encoder3 = FrnConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.encoder4 = FrnConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Up3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.decoder3 = FrnConv3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Up2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.decoder2 = FrnConv3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Up1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = FrnConv3d(2 * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv3d(n_filters, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, test=False):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        x = self.encoder4(self.pool(e3))
        x = self.decoder3(torch.cat([self.Up3(x), e3], 1))
        x = self.decoder2(torch.cat([self.Up2(x), e2], 1))
        x = self.decoder1(torch.cat([self.Up1(x), e1], 1))
        x = self.conv1x1(x)
        if test:
            return x
        x = projection(x)
        return x
