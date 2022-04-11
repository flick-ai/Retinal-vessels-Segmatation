# 函数实现文件，用力实现利用SSIM指标改进后的NLM滤波器
import cv2
import numpy as np
import matlab
import matlab.engine
import os


def create(Y, X):
    ave_y = np.mean(Y)
    mean_y = np.std(Y)
    ave_x = np.mean(X)
    mean_x = np.std(X)
    out = mean_x / mean_y * (Y - ave_y) + ave_x
    out = out.astype(int)
    return out


def SSIM(src, sample):
    # src = np.array(src/255, dtype=float)
    src = create(src, sample)
    f = 10
    t = 2
    H, W = src.shape
    sum_image = np.zeros((H, W), np.uint8)
    sum_weight = np.zeros((H, W), np.uint8)
    pad_length = f + t
    padding = cv2.copyMakeBorder(src, pad_length, pad_length, pad_length, pad_length, cv2.BORDER_REFLECT)
    image = padding[t:t + H + f + 1, t:t + W + f + 1]
    eng = matlab.engine.start_matlab()
    for r in range(-t, t):
        for s in range(-t, t):
            w_image = padding[t + r:t + H + f + r + 1, t + s:t + W + f + s + 1]
            weight = cal(image, w_image, eng)
            sum_image = sum_image + weight[f:f + H, f:f + W] ** 2 * w_image[f:f + H, f:f + W]
            sum_weight = sum_weight + weight[f:f + H, f:f + W] ** 2
    out = sum_image / sum_weight
    return out


def cal(img1, img2, eng):
    cv2.imwrite('2.jpg', img1)
    img1 = cv2.imread('2.jpg', 0)
    img1 = matlab.uint8(img1.tolist())
    cv2.imwrite('3.jpg', img2)
    img2 = cv2.imread('3.jpg', 0)
    os.remove('2.jpg')
    os.remove('3.jpg')
    img2 = matlab.uint8(img2.tolist())
    [ssim, ssim_map] = eng.ssim(img1, img2, nargout=2)
    return np.array(ssim_map)
