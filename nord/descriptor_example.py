"""
Created on 2018-08-12

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
from neural_nets import NeuralDescriptor, NeuralEvaluator
import neural_nets.layers.layer_types
import torch.nn as nn
import torch

input_sz = 32

conv = nn.Conv2d
conv_params = [3, 10, 5]

pool = nn.MaxPool2d
pool_params = [2, 2]

linear = nn.Linear
linear_params = [140*14, 10]

Flatten = neural_nets.layers.layer_types.Flatten
d = NeuralDescriptor()

d.add_layer(conv, conv_params, 'conv')
d.add_layer(pool, pool_params, 'pool1')
d.add_layer(pool, pool_params, 'pool2')

d.add_layer(Flatten, [], 'flat1')
d.add_layer(Flatten, [], 'flat2')

d.add_layer(linear, linear_params, 'lin')

d.connect_layers('conv', 'pool1')
d.connect_layers('conv', 'pool2')

d.connect_layers('pool1', 'flat1')
d.connect_layers('pool2', 'flat2')

d.connect_layers('flat1', 'lin')
d.connect_layers('flat2', 'lin')


evaluator = NeuralEvaluator()

sample = torch.Tensor(
    evaluator.get_sample()).transpose(2, 0)
sample = sample.unsqueeze(0)
evaluator.descriptor_evaluate(d, verbose=True)
