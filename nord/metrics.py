import torch


def extract_value(metric):
    return metric.float().mean().cpu().item()


def accuracy(predicted, targets):

    acc = extract_value(predicted.max(dim=1)[1] == targets)
    return {'accuracy': acc}


def one_hot_accuracy(predicted, targets):

    acc = extract_value(predicted.max(dim=1)[1] == targets.max(
        dim=1)[1])
    return {'accuracy': acc}
