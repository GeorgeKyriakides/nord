import torch.nn as nn


def My_CrossEntropyLoss():
    def mrl(outputs, labels):
        criterion_loss = nn.CrossEntropyLoss()
        outs = outputs.to(labels.device)
        targets = labels.to(
            labels.device).max(dim=1)[1]
        loss = criterion_loss(outs, targets)
        return loss
    return mrl
