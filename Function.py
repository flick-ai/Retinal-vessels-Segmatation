import visvis as vv
import numpy as np
import Args


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def show3D(vols):
    vols = [vols.transpose(1, 0, 2)[150:450, :, :]]
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
