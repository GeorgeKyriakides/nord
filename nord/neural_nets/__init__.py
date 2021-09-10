
"""
Created on Sun Aug  5 18:54:54 2018

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""


from . import layers
from .benchmark_evaluators import BenchmarkEvaluator
from .neural_builder import NeuralNet
from .neural_descriptor import NeuralDescriptor
from .neural_evaluators import (DistributedEvaluator, LocalBatchEvaluator,
                                LocalEvaluator)

__all__ = ['layers',
           'NeuralNet',
           'LocalEvaluator',
           'LocalBatchEvaluator',
           'NeuralDescriptor',
           'DistributedEvaluator',
           'data_curators',
           'BenchmarkEvaluator',
           ]
