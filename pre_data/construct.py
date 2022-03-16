# 子控制文件，用于导入数据集并生成对应的3D数据以及投影之后的2D数据
import numpy as np
import cv2
import os
from NLM import NLM
from SSIM import SSIM
import sys
sys.path.append("..")
import Args


def project():
    # 请使用者在此处添加你所使用的数据集的绝对路径
    path = Args.OCTA_path
    # 请使用者在此处添加你所使用的数据集的保存路径
    save_path = Args.OCTA_3D
    pack_img = np.arange(10001, 10301, 1)
    div_img = np.arange(1, 401, 1)
    for j in pack_img:
        sum_img = []
        for i in div_img:
            full_path = path + '/' + str(j) + '/' + str(i) + '.bmp'
            print(full_path)
            sum_img.append(cv2.imread(full_path, 0))
        sum_img = np.array(sum_img)
        # 保存三维数据以用来投影
        np.save(save_path + '/3D/' + str(j) + '.npy', sum_img)
        projection = np.sum(sum_img, axis=1)
        # 将投影转换至uint8
        projection = projection / np.max(projection)
        projection = np.clip(projection, 0, 1)
        projection = np.uint8(projection * 255)

        # 将投影做180度上下翻转至GroundTruth对应空间
        projection = np.flip(projection, 0)
        print(np.max(projection))
        # 保存原始投影图像
        cv2.imwrite(save_path + "/2D/" + str(j) + ".bmp", projection)
        # /print(os.path.exists(save_path + "/2D"))
        projection_nlm = NLM(projection)
        # 保存NLM滤波后投影图像
        cv2.imwrite(save_path + "/2D_NLM/" + str(j) + ".bmp", projection_nlm)
        # projection_ssim = SSIM(projection, projection_nlm)
        # 保存SSIM滤波后投影图像
        # cv2.imwrite(save_path + "2D_SSIM/" + str(j) + ".bmp", projection_ssim)


if __name__ == "__main__":
    project()
