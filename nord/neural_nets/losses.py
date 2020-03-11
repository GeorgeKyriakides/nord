import torch
from torch import Tensor
import torch.nn as nn


class SPCELoss(nn.CrossEntropyLoss):
    def __init__(self, *args, n_samples=0, **kwargs):
        super(SPCELoss, self).__init__(*args, **kwargs)
        self.threshold = 0.1
        self.growing_factor = 1.3
        self.initial_train = True
        # self.v = torch.zeros(n_samples).int()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        super_loss = nn.functional.cross_entropy(
            input, target, reduction="none")
        v = self.spl_loss(super_loss)
        # self.v[index] = v
        print(super_loss.type(), v.type())
        return (super_loss * v).mean()

    def increase_threshold(self):
        self.initial_train = False
        self.threshold *= self.growing_factor

    def spl_loss(self, super_loss):
        if self.initial_train:
            v = super_loss < 1e10
        else:
            v = super_loss < self.threshold
        # return v.int()
        return v.float()
