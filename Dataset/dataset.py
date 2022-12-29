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


def cut(data, label, step=64, standard=(256, 256, 192)):
    need_to_pad = np.array(standard) - np.array(data.shape)
    lbx = need_to_pad[0] // 2
    ubx = need_to_pad[0] // 2 + need_to_pad[0] % 2
    lby = need_to_pad[1] // 2
    uby = need_to_pad[1] // 2 + need_to_pad[1] % 2
    lbz = need_to_pad[2] // 2
    ubz = need_to_pad[2] // 2 + need_to_pad[2] % 2

    label = torch.reshape(label, data.shape)
    data = torch.nn.functional.pad(data, [lbz, ubz, lby, uby, lbx, ubx])
    label = torch.nn.functional.pad(label, [lbz, ubz, lby, uby, lbx, ubx])
    data, label = data.unsqueeze(dim=0), label.unsqueeze(dim=0)
    out, target = [], []
    for i in range(3):
        for j in range(3):
            for k in range(2):
                slice_data = data[:, i * step:(i + 1) * step, j * step:(j + 1) * step, k * step:(k + 1) * step]
                slice_target = label[:, i * step:(i + 1) * step, j * step:(j + 1) * step, k * step:(k + 1) * step]
                slice_target = one_hot(slice_target, 5, dim=0)
                if torch.max(slice_target) > 0:
                    out.append(slice_data)
                    target.append(slice_target)
    return out, target


def projection(data, label, standard=(256, 256, 192)):
    need_to_pad = np.array(standard) - np.array(data.shape)
    lbx = need_to_pad[0] // 2
    ubx = need_to_pad[0] // 2 + need_to_pad[0] % 2
    lby = need_to_pad[1] // 2
    uby = need_to_pad[1] // 2 + need_to_pad[1] % 2
    lbz = need_to_pad[2] // 2
    ubz = need_to_pad[2] // 2 + need_to_pad[2] % 2

    label = torch.reshape(label, data.shape)
    data = torch.nn.functional.pad(data, [lbz, ubz, lby, uby, lbx, ubx])
    label = torch.nn.functional.pad(label, [lbz, ubz, lby, uby, lbx, ubx])

    x_sum = torch.sum(data, dim=2)
    label_sum = torch.sum(label, dim=2)
    out = 255 * (x_sum - torch.min(x_sum)) / (torch.max(x_sum) - torch.min(x_sum))
    target = 255 * (label_sum - torch.min(label_sum)) / (torch.max(label_sum) - torch.min(label_sum))

    out = [out.unsqueeze(dim=0)]
    target = [target.unsqueeze(dim=0)]
    return out, target


class MyDataset(Dataset):
    def __init__(self, folder, transform=None, target_transform=None, loader=default_loader,
                 target_loader=default_loader, valid=False):
        super(MyDataset, self).__init__()
        subdirs = [x for x in os.listdir(folder) if not os.path.isdir(x)]
        self.data = []
        for p in subdirs:
            patdir = join(folder, p)
            t1 = join(patdir, p + "_t1.nii")
            # my addition
            t2 = join(patdir, p + "_t2.nii")
            t1ce = join(patdir, p + "_t1ce.nii")
            flair = join(patdir, p + "_flair.nii")

            seg = join(patdir, p + "_seg.nii")

            self.data.append([t1, t2, t1ce, flair, seg])

        if valid:
            self.data = random.sample(self.data, int(0.2 * len(self.data)))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.target_loader = target_loader

    # The original __getitem__ function

    # def __getitem__(self, index):
    #     fn, label = self.data[index]
    #     img = nib.load(fn).get_fdata()
    #     target = nib.load(label).get_fdata()
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #     img, target = cut(img, target)
    #     return img, target

    def __getitem__(self, index):
        fn, fn2, fn3, fn4, label = self.data[index]
        img = nib.load(fn).get_fdata()    #t1
        img2 = nib.load(fn2).get_fdata()    #t2
        img3 = nib.load(fn3).get_fdata()    #t1ce
        img4 = nib.load(fn4).get_fdata()    #flair
        target = nib.load(label).get_fdata()    #seg
        
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
        if self.target_transform is not None:
            target = self.target_transform(target)
        #My addition
        target0 = target
        img, target = projection(img, target0)
        img2, target = projection(img2, target0)
        img3, target = projection(img3, target0)
        img4, target = projection(img4, target0)

        return img, img2, img3, img4, target

    def __len__(self):
        return len(self.data)


# target0 = target
# img, target = cut(img, target0)
# img2, target = cut(img2, target0)
# img3, target = cut(img3, target0)
# img4, target = cut(img4, target0)