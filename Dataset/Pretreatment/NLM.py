#  函数实现文件，实现了NLM滤波器作为预处理部分
import numpy as np


def NLM(src, f=2, t=5, h=10):
    # src = np.array(src/255, dtype=float)
    H, W = src.shape
    sumimage = np.zeros((H, W), np.uint8)
    sumweight = np.zeros((H, W), np.uint8)
    pad_length = f + t
    padding = np.pad(src, (pad_length, pad_length), mode='symmetric').astype(np.float32)
    image = padding[t:t + H + f + 1, t:t + W + f + 1]
    M, N = image.shape
    h = h * h
    maxweight = np.zeros((H, W), np.uint8)
    for r in range(-t, t):
        for s in range(-t, t):
            if r == 0 and s == 0:
                continue;
            wimage = padding[t + r:t + H + f + r + 1, t + s:t + W + f + s + 1]
            diff = (image - wimage) * (image - wimage)
            J = np.cumsum(diff, 0)
            J = np.cumsum(J, 1)
            distance = J[M - H:M + 1, N - W:N + 1] + J[0:H, 0:W] - J[M - H:M + 1, 0:W] - J[0:H, N - W:N + 1]
            distance = distance / (2 * f + 1) / (2 * f + 1)
            weight = np.exp(-distance / h)
            sumimage = sumimage + weight * wimage[f:f + H, f:f + W]
            sumweight = sumweight + weight
            if weight.any() > maxweight.any():
                weight_max = maxweight
    sumimage = sumimage + maxweight * image[f:f + H, f:f + W]
    sumweight = sumweight + weight_max
    out = sumimage / sumweight

    # if out.min() < 0:
    #     low_clip = -1.
    # else:
    #     low_clip = 0.
    # out = np.clip(out, 0, 1.0)
    # out = np.uint8(out * 255)

    return out
