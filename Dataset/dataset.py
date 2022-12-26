# 子控制文件，负责将指定文件地址的数据集转换为可训练形式
import ipdb
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from os.path import join
import os
import nibabel as nib
import random
from tqdm import tqdm
from Function import show3D
from monai.networks import one_hot


def default_loader(path):
    return Image.open(path)


def numpy_loader(path):
    return np.load(path)


def cut(data, label, step=64, standard=(192, 256, 256)):
    need_to_pad = np.array(standard) - np.array(data.shape)
    lbx = need_to_pad[0] // 2
    ubx = need_to_pad[0] // 2 + need_to_pad[0] % 2
    lby = need_to_pad[1] // 2
    uby = need_to_pad[1] // 2 + need_to_pad[1] % 2
    lbz = need_to_pad[2] // 2
    ubz = need_to_pad[2] // 2 + need_to_pad[2] % 2

    data = torch.nn.functional.pad(data, [lbz, ubz, lby, uby, lbx, ubx])
    label = torch.nn.functional.pad(label, [lbz, ubz, lby, uby, lbx, ubx])
    out = data[48:144, 64:192, 64:192]
    target = label[48:144, 64:192, 64:192]
    # data, label = data.unsqueeze(dim=0), label.unsqueeze(dim=0)
    # out, target = [], []
    # for i in range(3):
    #     for j in range(3):
    #         for k in range(2):
    #             slice_data = data[:, i * step:(i + 1) * step, j * step:(j + 1) * step, k * step:(k + 1) * step]
    #             slice_target = label[:, i * step:(i + 1) * step, j * step:(j + 1) * step, k * step:(k + 1) * step]
    #             # slice_target = one_hot(slice_target, 5, dim=0)
    #             if torch.max(slice_data) > 0 and torch.max(slice_target) > 0:
    #                 out.append(slice_data)
    #                 target.append(slice_target)
    out, target = out.unsqueeze(dim=0), target.unsqueeze(dim=0)
    # print(out.shape, target.shape)
    return [out], [target]


class MyDataset(Dataset):
    def __init__(self, folder, transform=None, target_transform=None, loader=default_loader,
                 target_loader=default_loader, valid=False):
        super(MyDataset, self).__init__()
        subdirs = [x for x in os.listdir(folder) if not os.path.isdir(x)]
        self.data = []
        for p in subdirs:
            patdir = join(folder, p)
            t1 = join(patdir, p + "_t1.nii")
            seg = join(patdir, p + "_seg.nii")
            self.data.append([t1, seg])

        if valid:
            self.data = random.sample(self.data, int(0.2 * len(self.data)))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.target_loader = target_loader

    def __getitem__(self, index):
        fn, label = self.data[index]
        img = nib.load(fn).get_fdata()
        target = nib.load(label).get_fdata()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        img, target = cut(img, target)
        return img, target

    def __len__(self):
        return len(self.data)
