import ipdb
import numpy as np
from tqdm import tqdm
from Function import count_parameters, Dice, show3D
import torch
from monai.metrics import compute_generalized_dice
from monai.networks import one_hot


class Train:
    def __init__(self, data, val, device, model, loss, optimizer, time, split, path):
        self.device = device
        self.model = model
        self.data = data
        self.function = loss
        self.optimizer = optimizer
        self.path = path
        self.epoch = time
        self.val = val
        self.max = 0

    def Save(self):
        loss = []
        with torch.no_grad():
            for batch_idx, (list_data, list_label) in tqdm(enumerate(self.val), total=len(self.val)):
                self.model.eval()
                for data, target in zip(list_data, list_label):
                    data, target = data.float(), target.float()
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss.append(Dice(output, target))
                    # loss.append(compute_generalized_dice(output, one_hot(target, 5, dim=1),include_background=False))
        result = sum(loss) / len(loss)
        print("Test Dice:{}".format(result))
        if result > self.max:
            self.max = result
            print("Save successfully!")
            torch.save(self.model, self.path)

    def train_one(self):
        self.model.train()
        sum_loss = 0
        for batch_idx, (list_data, list_label) in tqdm(enumerate(self.data), total=len(self.data)):
            data_loss = 0
            for data, target in zip(list_data, list_label):
                data, target = data.float(), target.float()
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.function(output, target)
                data_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            sum_loss = sum_loss + data_loss / len(list_data)
        result = sum_loss / len(self.data)
        print("Train loss:{}".format(result))
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
        loss = []
        with torch.no_grad():
            for batch_idx, (list_data, list_label) in tqdm(enumerate(self.data), total=len(self.data)):
                self.model.eval()
                for data, target in zip(list_data, list_label):
                    data, target = data.float(), target.float()
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    data = data[0, 0, :, :, :].cpu().numpy()
                    # target = torch.topk(target, 1, dim=1)[1][0, 0, :, :, :].cpu().numpy()
                    # output = torch.topk(output, 1, dim=1)[1][0, 0, :, :, :].cpu().numpy()
                    target = target[0, 0, :, :, :].cpu().numpy()
                    output = output[0, 0, :, :, :].cpu().numpy()
                    show3D(data)
                    show3D(target)
                    show3D(output)
            print(sum(loss) / len(loss))
