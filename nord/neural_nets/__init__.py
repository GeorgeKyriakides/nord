
"""
Created on Sun Aug  5 18:54:54 2018

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""


from .evaluators.nasbench101_evaluator import NASBench_101Evaluator
from .evaluators.natsbench_evaluator import NATSBench_Evaluator
from .evaluators.neural_evaluators import DistributedEvaluator, LocalEvaluator
from .neural_builder import NeuralNet
from .neural_descriptor import NeuralDescriptor

__all__ = ['NeuralNet',
           'LocalEvaluator',
           'NeuralDescriptor',
           'DistributedEvaluator',
           'data_curators',
           'NASBench_101Evaluator',
           'NATSBench_Evaluator',
           ]
