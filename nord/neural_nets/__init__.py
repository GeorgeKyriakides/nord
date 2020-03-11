"""
Created on 2018-08-05

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

from .layers import *
from .local.data_curator import get_cifar10
from .distributed.data_curator import get_cifar10_distributed
from .neural_builder import NeuralNet
from .local.neural_evaluator import NeuralEvaluator
from .distributed.neural_evaluator import DistributedNeuralEvaluator
from .neural_descriptor import NeuralDescriptor


__all__ = ['layers',
           'NeuralNet',
           'NeuralEvaluator',
           'NeuralDescriptor',
           'DistributedNeuralEvaluator',
           'get_cifar10',
           'get_cifar10_distributed']
