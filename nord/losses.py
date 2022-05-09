import torch.nn as nn
from nord.utils import create_pairs


def nord_cross_entropy_loss():
    def mrl(outputs, labels):
        criterion_loss = nn.CrossEntropyLoss()
        outs = outputs.to(labels.device)
        targets = labels.to(
            labels.device).max(dim=1)[1]
        loss = criterion_loss(outs, targets)
        return loss
    return mrl


def nord_margin_rank_loss():
    def mrl(outputs, labels):
        criterion_loss = nn.MarginRankingLoss()
        x, y, z = create_pairs(outputs, labels.float())
        loss = criterion_loss(x.to(labels.device), y.to(
            labels.device), z.to(labels.device))
        return loss
    return mrl

# class SPCELoss(nn.CrossEntropyLoss):
#     def __init__(self, *args, n_samples=0, **kwargs):
#         super(SPCELoss, self).__init__(*args, **kwargs)
#         self.threshold = 0.1
#         self.growing_factor = 1.3
#         self.initial_train = True
#         # self.v = torch.zeros(n_samples).int()

#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         super_loss = nn.functional.cross_entropy(
#             input, target, reduction="none")
#         v = self.spl_loss(super_loss)
#         # self.v[index] = v
#         print(super_loss.type(), v.type())
#         return (super_loss * v).mean()

#     def increase_threshold(self):
#         self.initial_train = False
#         self.threshold *= self.growing_factor

#     def spl_loss(self, super_loss):
#         if self.initial_train:
#             v = super_loss < 1e10
#         else:
#             v = super_loss < self.threshold
#         # return v.int()
#         return v.float()
