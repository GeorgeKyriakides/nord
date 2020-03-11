
"""
Created on Sun Aug  5 18:54:54 2018

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

import neural_nets.layers as layers
from .data_curators import get_cifar10
from .data_curators import get_cifar10_distributed
from .neural_builder import NeuralNet
from .neural_evaluators import LocalEvaluator
from .neural_evaluators import DistributedEvaluator
from .neural_descriptor import NeuralDescriptor
import neural_nets.distributed_partial_sampler as distributed_partial_sampler


__all__ = ['layers',
           'NeuralNet',
           'LocalEvaluator',
           'NeuralDescriptor',
           'DistributedEvaluator',
           'get_cifar10',
           'get_cifar10_distributed',
           'distributed_partial_sampler']
