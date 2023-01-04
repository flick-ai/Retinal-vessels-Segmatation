from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

with open("G:/term5/BI_proj/Proj/my-BraTS2020/NetSave/Old/loss.txt", "r", encoding='utf-8') as f:  #打开文本
    data = f.read()   #读取文本
data = data.split("\n")
data = data[:-1]

for i in range(len(data)):
    data[i] = float(data[i])

x_index = []
index = 0
for i in range(len(data)):
    x_index.append(index)
    index += 1

plt.plot(x_index, data)
plt.xlabel("epoch")
plt.ylabel("loss")
fig = plt.gcf()
plt.show()
fig.savefig("G:/term5/BI_proj/Proj/my-BraTS2020/NetSave/Old/loss.png")