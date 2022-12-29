import ipdb
import torch
from torch import nn
from torch.nn import functional as F
from .layers import BasicConv3d, BasicConv2d, Upconv2d, projection, AttConv3d, SEConv3d, AttConv2d, SEConv2d
import cv2
from axial_attention import AxialAttention
from Function import show3D
import numpy as np

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

    def forward(self, x, test=False):
        e1 = self.Conv1(x)
        e2 = self.Conv2(self.pool(e1))
        e3 = self.Conv3(self.pool(e2))
        e4 = self.Conv4(self.pool(e3))
        x = self.Conv5(self.pool(e4))
        x = self.decoder4(torch.cat((self.Up4(x), e4), 1))
        x = self.decoder3(torch.cat((self.Up3(x), e3), 1))
        x = self.decoder2(torch.cat((self.Up2(x), e2), 1))
        x = self.decoder1(torch.cat((self.Up1(x), e1), 1))
        # x = self.conv1x1(x)
        return x

# Because now I need to use 2D, so I use 2D to replace 3D

class BaselineUnet(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters):    # (1, 1, 8)
        super(BaselineUnet, self).__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.out_channels = out_channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1 = BasicConv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.encoder2 = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.encoder3 = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.encoder4 = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.Up3 = nn.ConvTranspose2d(32 * n_filters, 16 * n_filters, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.decoder3 = BasicConv2d(32 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Up2 = nn.ConvTranspose2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.decoder2 = BasicConv2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Up1 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = BasicConv2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(4 * n_filters, out_channels, kernel_size=1, stride=1, padding=0)

    def Encoder(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        return e1, e2, e3, e4
    def Decoder(self, total_e1, total_e2, total_e3, total_e4):
        x = self.decoder3(torch.cat([self.Up3(total_e4), total_e3], 1))
        x = self.decoder2(torch.cat((self.Up2(x), total_e2), 1))
        x = self.decoder1(torch.cat((self.Up1(x), total_e1), 1))
        x = self.conv1x1(x)
        return x
    
    # I add my additional input here
    def forward(self, x0,x1,x2,x3, test=False):
        e0_1, e0_2, e0_3, e0_4 = self.Encoder(x0)
        e1_1, e1_2, e1_3, e1_4 = self.Encoder(x1)
        e2_1, e2_2, e2_3, e2_4 = self.Encoder(x2)
        e3_1, e3_2, e3_3, e3_4 = self.Encoder(x3)

        total_e1 = torch.cat([e0_1, e1_1, e2_1, e3_1], 1)
        total_e2 = torch.cat([e0_2, e1_2, e2_2, e3_2], 1)
        total_e3 = torch.cat([e0_3, e1_3, e2_3, e3_3], 1)
        total_e4 = torch.cat([e0_4, e1_4, e2_4, e3_4], 1)

        x = self.Decoder(total_e1, total_e2, total_e3, total_e4)
        if test:
            return x
        # x = projection(x)
        return x


class AttentionUnet(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters):
        super(AttentionUnet, self).__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.out_channels = out_channels

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder1 = AttConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.encoder2 = AttConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.encoder3 = AttConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.encoder4 = AttConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.attention = AxialAttention(dim=36, dim_index=2, heads=4, num_dimensions=3)
        self.Up3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.decoder3 = AttConv3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Up2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.decoder2 = AttConv3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Up1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = AttConv3d(2 * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv3d(n_filters, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, test=False):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        x = self.encoder4(self.pool(e3))
        device = x.device
        self.attention = self.attention.to(device)
        # print(x.shape)
        att = self.attention(x)
        x = torch.mul(att, x)
        x = self.decoder3(torch.cat([self.Up3(x), e3], 1))
        x = self.decoder2(torch.cat([self.Up2(x), e2], 1))
        x = self.decoder1(torch.cat([self.Up1(x), e1], 1))
        x = self.conv1x1(x)
        return x


class SeUnet(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters):
        super(SeUnet, self).__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.out_channels = out_channels
        reduction = 2

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder1 = SEConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.encoder2 = SEConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.encoder3 = SEConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.encoder4 = SEConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Up3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.decoder3 = SEConv3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Up2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.decoder2 = SEConv3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.Up1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = SEConv3d(2 * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
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
        return x


# class BaselineUnet(nn.Module):
#     def __init__(self, in_channels, out_channels, n_filters):    # (1, 1, 8)
#         super(BaselineUnet, self).__init__()
#         self.in_channels = in_channels
#         self.n_filters = n_filters
#         self.out_channels = out_channels

#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder1 = BasicConv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
#         self.encoder2 = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
#         self.encoder3 = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
#         self.encoder4 = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

#         self.Attencoder1 = AttConv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
#         self.Attencoder2 = AttConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
#         self.Attencoder3 = AttConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
#         self.Attencoder4 = AttConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
#         self.attention = AxialAttention(dim=36, dim_index=2, heads=4, num_dimensions=3)

#         self.SeUencoder1 = SEConv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
#         self.SeUencoder2 = SEConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
#         self.SeUencoder3 = SEConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
#         self.SeUencoder4 = SEConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

#         self.Up3 = nn.ConvTranspose2d(32 * n_filters, 16 * n_filters, kernel_size=3, stride=2, padding=1,
#                                       output_padding=1)
#         self.decoder3 = BasicConv2d(32 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
#         self.Up2 = nn.ConvTranspose2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1,
#                                       output_padding=1)
#         self.decoder2 = BasicConv2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
#         self.Up1 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.decoder1 = BasicConv2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
#         self.conv1x1 = nn.Conv2d(4 * n_filters, out_channels, kernel_size=1, stride=1, padding=0)





    
#     # I add my additional input here
#     def forward(self, x0,x1,x2,x3, test=False):
#         e1 = self.encoder1(x0)
#         e2 = self.encoder2(self.pool(e1))
#         e3 = self.encoder3(self.pool(e2))
#         e4 = self.encoder4(self.pool(e3))

#         ne1 = self.encoder1(x3)
#         ne2 = self.encoder2(self.pool(ne1))
#         ne3 = self.encoder3(self.pool(ne2))
#         ne4 = self.encoder4(self.pool(ne3))

#         Atte1 = self.Attencoder1(x1)
#         Atte2 = self.Attencoder2(self.pool(Atte1))
#         Atte3 = self.Attencoder3(self.pool(Atte2))
#         Atte4 = self.Attencoder4(self.pool(Atte3))

#         SeUe1 = self.SeUencoder1(x2)
#         SeUe2 = self.SeUencoder2(self.pool(SeUe1))
#         SeUe3 = self.SeUencoder3(self.pool(SeUe2))
#         SeUe4 = self.SeUencoder4(self.pool(SeUe3))

#         # x = self.decoder4(torch.cat([self.Up4(x), e4], 1))
#         totale4 = torch.cat([e4,ne4,Atte4,SeUe4],1)
#         totale3 = torch.cat([e3,ne3,Atte3,SeUe3],1)
#         totale2 = torch.cat([e2,ne2,Atte2,SeUe2],1)
#         totale1 = torch.cat([e1,ne1,Atte1,SeUe1],1)
        
#         x = self.decoder3(torch.cat([self.Up3(totale4), totale3], 1))
#         x = self.decoder2(torch.cat([self.Up2(x), totale2], 1))
#         x = self.decoder1(torch.cat([self.Up1(x), totale1], 1))
#         x = self.conv1x1(x)
#         if test:
#             return x
#         # x = projection(x)
#         return x

