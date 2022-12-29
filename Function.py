import visvis as vv
import numpy as np
import torch
from PIL import Image


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def show3D(vols):
    vols = [vols.transpose(1, 0, 2)]
    f = vv.clf()
    a = vv.gca()
    m = vv.MotionDataContainer(a)
    for vol in vols:
        t = vv.volshow(vol)
        t.parent = m
        t.colormap = vv.ColormapEditor
    a.daspect = 1, 1, -1
    a.xLabel = 'x'
    a.yLabel = 'y'
    a.zLabel = 'z'
    app = vv.use()
    app.Run()

def show2D(vols, target):
    im = Image.fromarray(np.uint8(vols))
    tar = Image.fromarray(np.uint8(target))
    im.save('test.png')
    tar.save('target.png')


def read3D(path):
    data = np.load(path)
    data = data.transpose(1, 0, 2)
    data = data.reshape((1, 288, 400, 400))
    x_sum = np.sum(data, 1)
    out = 255 * (x_sum - np.min(x_sum)) / (np.max(x_sum) - np.min(x_sum))
    return data


def Dice(output, target):
    smooth = 1
    # axes = tuple(range(1, output.dim()))
    intersect = torch.sum(output * target)
    union = torch.sum(torch.pow(output, 2)) + torch.sum(torch.pow(target, 2))
    loss = (2 * intersect + smooth) / (union + smooth)
    return loss.mean()

