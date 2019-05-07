import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, prediction, target):
        return torch.sqrt(self.mse(prediction, target) + self.eps)


class GraphicLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        add_grid = torch.add(prediction.t(), target)
        return torch.sum(add_grid == 2.).float()


def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)


def soft_cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

