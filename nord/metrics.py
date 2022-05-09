import numpy as np
import torch
from nord.utils import rank_inds_of_lists
from scipy.stats import kendalltau, rankdata, spearmanr


def extract_value(metric):
    return metric.float().mean().cpu().item()


def accuracy(predicted, targets):

    acc = extract_value(predicted.max(dim=1)[1] == targets)
    return {'accuracy': acc}


def MSE(predicted, targets):
    # if type(predicted) == list or type(predicted) == tuple:
    #     predicted = torch.stack(predicted)
    #     targets = torch.stack(targets)
    mse = torch.nn.functional.mse_loss(predicted, targets).float().cpu()
    return {'mse': mse}


def one_hot_accuracy(predicted, targets):

    acc = extract_value(predicted.max(dim=1)[1] == targets.max(
        dim=1)[1])
    return {'accuracy': acc}


def binary_rank_correlation_tau_with_top(p, metric='kendall'):
    chosen_metric = kendalltau
    if metric == 'spearman':
        chosen_metric = spearmanr

    def ktwt(predicted, targets, all_to_all=True):
        # if type(predicted) == list or type(predicted) == tuple:
        #     predicted = torch.stack(predicted)
        #     targets = torch.stack(targets)

        original_ranks = rank_inds_of_lists(targets, all_to_all=all_to_all)
        calculated_ranks = rank_inds_of_lists(
            predicted, all_to_all=all_to_all)

        all_t = chosen_metric(original_ranks, calculated_ranks)

        sz = len(original_ranks)
        top_half_inds = np.argsort(original_ranks)[-int(sz*p):]
        original_top_ranks = rank_inds_of_lists(
            targets, indices=top_half_inds, all_to_all=all_to_all)
        calculated_top_ranks = rank_inds_of_lists(
            predicted, indices=top_half_inds, all_to_all=all_to_all)
        top_t = chosen_metric(original_top_ranks,
                              calculated_top_ranks, all_to_all)

        return {metric: all_t, 'top_'+metric: top_t}

    return ktwt


def regression_rank_correlation_with_top(p, metric='kendall'):
    chosen_metric = kendalltau
    if metric == 'spearman':
        chosen_metric = spearmanr

    def ktwt(predicted, targets, all_to_all=True):
        # if type(predicted) == list or type(predicted) == tuple:
        #     predicted = torch.stack(predicted)
        #     targets = torch.stack(targets)
        original_ranks = rankdata((targets.cpu().reshape((-1, ))*3333).round())
        calculated_ranks = rankdata(
            (predicted.cpu().reshape((-1, ))*3333).round())

        all_t = chosen_metric(original_ranks, calculated_ranks)

        sz = len(original_ranks)
        top_half_inds = np.argsort(original_ranks)[-int(sz*p):]
        original_top_ranks = rankdata(
            targets.cpu().reshape((-1, ))*3333).round()[top_half_inds]
        calculated_top_ranks = rankdata(
            predicted.cpu().reshape((-1, ))*3333).round()[top_half_inds]
        top_t = chosen_metric(original_top_ranks, calculated_top_ranks)

        return {metric: all_t, 'top_'+metric: top_t}

    return ktwt
