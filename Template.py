import numpy as np
from tqdm import tqdm
from Function import count_parameters
from model.layers import projection
from torch import nn
import torch
import cv2
from Function import show3D


class Train:
    def __init__(self, data, val, device, model, loss, optimizer, time, path):
        self.device = device
        self.model = model
        self.data = data
        self.function = loss
        self.optimizer = optimizer
        self.path = path + self.__class__.__name__ + "1.pth"
        self.epoch = time
        self.val = val
        self.max = 0

    def Save(self):
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(self.val), total=len(self.val)):
                self.model.eval()
                data, target = data.float(), target.float()
                data0, target0 = data.to(self.device), target.to(self.device)
                output = np.zeros([400, 400])
                for i in range(5):
                    for j in range(5):
                        data = data0[:, :, :, i * 68:i * 68 + 128, j * 68:j * 68 + 128]
                        target = target0[:, :, i * 68:i * 68 + 128, j * 68:j * 68 + 128]
                        out = self.model(data)
                        out = (out.cpu()).detach().numpy()
                        output[i * 68:i * 68 + 128, j * 68:j * 68 + 128] = out[0, 0, :, :]
                # cv2.imwrite(str(batch_idx)+'.png', output)
        print("Save successfully!")
        torch.save(self.model, self.path)

    def train_one(self):
        self.model.train()
        sum_loss = 0
        for batch_idx, (data, target) in tqdm(enumerate(self.data), total=len(self.data)):
            data, target = data.float(), target.float()
            data0, target0 = data.to(self.device), target.to(self.device)
            for i in range(5):
                for j in range(5):
                    data = data0[:, :, :, i * 68:i * 68 + 128, j * 68:j * 68 + 128]
                    target = target0[:, :, i * 68:i * 68 + 128, j * 68:j * 68 + 128]
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.function(output, target)
                    sum_loss = sum_loss + loss
                    loss.backward()
                    self.optimizer.step()
        print(sum_loss / len(self.data))
        self.Save()
        return sum_loss

    def train(self):
        print("Begin training:")
        print("The computing device:", "GPU" if self.device.type == "cuda" else "CPU")
        print("Total number of parameters:{}".format(str(count_parameters(self.model))))
        for i in range(self.epoch):
            self.train_one()


class Test:
    def __init__(self, data, device, model, path):
        self.device = device
        self.model = model
        self.data = data
        self.path = path

    def test(self):
        print("Begin testing:")
        print("The computing device:", "GPU" if self.device.type == "cuda" else "CPU")
        print("Total number of parameters:{}".format(str(count_parameters(self.model))))
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(self.data), total=len(self.data)):
                self.model.eval()
                data, target = data.float(), target.float()
                data0, target0 = data.to(self.device), target.to(self.device)
                output = np.zeros([288, 400, 400])
                for i in range(5):
                    for j in range(5):
                        data = data0[:, :, :, i * 68:i * 68 + 128, j * 68:j * 68 + 128]
                        # target = target0[:, :, i * 68:i * 68 + 128, j * 68:j * 68 + 128]
                        out = self.model(data, test=True)
                        out = (out.cpu()).detach().numpy()
                        output[:, i * 68:i * 68 + 128, j * 68:j * 68 + 128] = out[0, 0, :, :, :]
                show3D(output)
