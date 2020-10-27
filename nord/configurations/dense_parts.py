import torch
import torch.nn as nn


def get_dense_net(layers_number: int, output_size: int, activation: nn.Module):

    def dense_net(input_size):
        layers = {}
        for i in range(layers_number):
            layers['fc_%d' % i] = nn.Linear(
                input_size//pow(2, i), input_size//pow(2, i+1))
            layers['relu_%d' % i] = nn.ReLU()

        layers['fc_final'] = nn.Linear(input_size//pow(2, i+1), output_size)
        layers['activation_final'] = activation

        return layers

    return dense_net


def get_dense_batch_net(layers_number: int, output_size: int, activation: nn.Module):

    def dense_batch_net(input_size):
        layers = {}
        for i in range(layers_number):
            layers['fc_%d' % i] = nn.Linear(
                input_size//pow(2, i), input_size//pow(2, i+1))
            layers['relu_%d' % i] = nn.ReLU()
            layers['norm_%d' % i] = nn.BatchNorm1d(input_size//pow(2, i+1))

        layers['fc_final'] = nn.Linear(input_size//pow(2, i+1), output_size)
        layers['activation_final'] = activation

        return layers

    return dense_batch_net
