# 子控制文件，用于导入数据集并生成对应的3D数据以及投影之后的2D数据
import numpy as np
import cv2
from Pretreatment.NLM import NLM
from Pretreatment.SSIM import SSIM
import sys

sys.path.append("..")
import Args
from Function import show3D


def project():
    # 请使用者在此处添加你所使用的数据集的绝对路径
    path = Args.Dataset + '/OCTA/OCTA_6M_OCTA'
    # 请使用者在此处添加你所使用的数据集的保存路径
    save_path = Args.Dataset + '/Data'
    pack_img = np.arange(10001, 10301, 1)
    div_img = np.arange(1, 401, 1)
    for j in pack_img:
        sum_img = []
        for i in div_img:
            full_path = path + '/' + str(j) + '/' + str(i) + '.bmp'
            print(full_path)
            sum_img.append(cv2.imread(full_path, 0))
        sum_img = np.array(sum_img)
        sum_img = sum_img[:, 156:444, :]
        # show3D(sum_img)
        # 保存三维数据以用来投影
        sum_img = np.flip(sum_img, 0)
        projection = np.sum(sum_img, axis=1)
        # 将投影转换至uint8
        projection = (projection - np.min(projection)) / (np.max(projection)-np.min(projection))
        projection = np.uint8(projection * 255)
        if 10001 <= j <= 10180:
            np.save(save_path + '/3D/Train/' + str(j) + '.npy', sum_img)
            cv2.imwrite(save_path + "/2D/Train/" + str(j) + ".bmp", projection)
        elif 10181 <= j <= 10200:
            np.save(save_path + '/3D/Validation/' + str(j) + '.npy', sum_img)
            cv2.imwrite(save_path + "/2D/Validation/" + str(j) + ".bmp", projection)
        else:
            np.save(save_path + '/3D/Test/' + str(j) + '.npy', sum_img)
            cv2.imwrite(save_path + "/2D/Test/" + str(j) + ".bmp", projection)


if __name__ == "__main__":
    project()
