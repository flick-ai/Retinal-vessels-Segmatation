import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class softDiceloss(nn.Module):
    def __init__(self, wieght=None, size_average=True):
        super(softDiceloss, self).__init__()
    
    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        intersection = (m1*m2)
        score = 2.*(intersection.sum(1)+smooth)/ (m1.sum(1)+m2.sum(1)+smooth)
        score = 1-score.sum()/num
        return score
        