# 主控制文件，在此处可直接利用测试组测试网络性能
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import Args
import cv2
from mayavi import mlab


path = Args.OCTA_path + '/10001/'
div_img = np.arange(1, 401, 1)
img = []
for i in div_img:
    full_path = path + str(i) + '.bmp'
    # print(full_path)
    img.append(cv2.imread(full_path, 0))
img = np.array(img)
data_root_path = Args.Dataset + '/OCTA/train/target/10001.bmp'
target = cv2.imread(data_root_path, 0)
print(target)
img = img.transpose(0, 2, 1)
for i in range(640):
    img[:, :, i] = target * img[:, :, i]
mlab.contour3d(img, contours=5, transparent=False)
mlab.show()
