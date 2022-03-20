from tqdm import tqdm
from Function import count_parameters
from torch import nn
import torch


class Train:
    def __init__(self, data, device, model, loss, optimizer, time, path):
        self.device = device
        self.model = model
        self.data = data
        self.function = loss
        self.optimizer = optimizer
        self.path = path
        self.epoch = time

    def train(self):
        print("Begin training:")
        print("The computing device:", "GPU" if self.device.type == "cuda" else "CPU")
        print("Total number of parameters:{}".format(str(count_parameters(self.model))))
        for i in range(self.epoch):
            loss = self.control()
            print(loss)
        torch.save(self.model, self.path)

    def control(self):
        self.model.train()
        sum_loss = 0
        for batch_idx, (data, target) in tqdm(enumerate(self.data), total=len(self.data)):
            data, target = data.float(), target.float()
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.function(output, target)
            sum_loss = sum_loss + loss
            loss.backward()
            self.optimizer.step()
        return sum_loss


