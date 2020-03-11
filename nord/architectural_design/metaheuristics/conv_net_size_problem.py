"""
Created on 2018-08-02

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import torch.nn as nn
from neural_nets import NeuralDescriptor


class ConvNetSizeProblem:
    """A pygmo User Defined Problem, aimed at optimizing
    a ConvNet's layers size and channels.
    """

    def __init__(self, max_layers=2, distributed=False, untrained=False):
        """
        Parameters
        ----------
        max_layers : int
            The network's depth.

        """
        self.bounds = ([], [])
        self.layers = []
        self.data_channels = 3
        self.data_height = 32
        self.data_width = 32
        self.untrained = untrained
        self.distributed = distributed

        for _ in range(max_layers):
            self.bounds[0].append(1)  # Lower bound of layer's channels
            self.bounds[1].append(50)  # Upper bound of layer's channels
            self.bounds[0].append(1)  # Lower bound of layer's size
            self.bounds[1].append(50)  # Upper bound of layer's size

        if self.distributed:
            from neural_nets import DistributedNeuralEvaluator
            self.evaluator = DistributedNeuralEvaluator()
        else:
            from neural_nets import NeuralEvaluator
            self.evaluator = NeuralEvaluator()

    def get_bounds(self):
        """Returns the upper and lower limit for each layer's two parameters:
            -layer's channels and
            -layer's size



        Returns
        -------
        bounds : tuple
            A tuple of the requested parameters, in the form of
            ([cl_0, kl_0, cl_1, kl_1, ...], [cu_0, ku_0, cu_1, ku_1, ...] )

        """
        return self.bounds

    def get_nix(self):
        """Number of integer dimensions for the problem
        """
        return len(self.bounds[0])

    def fitness(self, solution):
        """Returns the solution's fitness (accuracy).

        Parameters
        ----------
        solution : list
            The network's parameters

        Returns
        -------
        fitness : float
            The network's accuracy, in the range [0.0, 100.0].

        """
        descriptor = NeuralDescriptor()
        previous_in = self.data_channels
        for i in range(0, len(solution), 2):
            out_channels = int(solution[i])
            kernel_size = int(solution[i+1])
            layer = nn.Conv2d
            parameters = (previous_in, out_channels, kernel_size)
            previous_in = out_channels
            descriptor.add_layer_sequential(layer, parameters)
            descriptor.add_layer_sequential(nn.ReLU6, [])
        fitness = self.evaluator.descriptor_evaluate(descriptor,
                                                     untrained=self.untrained,
                                                     verbose=False) * 100

        return [fitness]
