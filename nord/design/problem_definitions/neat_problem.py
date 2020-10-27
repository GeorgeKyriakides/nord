"""
Created on 2018-08-02

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import torch.nn as nn
from neural_nets import NeuralDescriptor


class DeepNeatProblem:
    """A pygmo User Defined Problem, aimed at optimizing
    a Net's layers types.
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
        self.max_layers = max_layers

        def set_bounds(min_, max_):
            self.bounds[0].append(min_)
            self.bounds[1].append(max_)

        # Floats
        for _ in range(max_layers):
            # Lower/upper bound of layer's dropout rate
            set_bounds(0.0, 0.7)
        for _ in range(max_layers):
            # Lower/upper bound of layer's weight scaling
            set_bounds(0.0, 1.0)
        # Integers
        for _ in range(max_layers):
            # Lower/upper bound of layer's filter no
            set_bounds(32, 256)
        for _ in range(max_layers):
            # Lower/upper bound of layer's kernel size
            set_bounds(1, 2)
        for _ in range(max_layers):
            # Lower/upper bound of layer's max pooling (T/F)
            set_bounds(0, 1)

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
        return self.max_layers * 3

    def fitness(self, solution):
        """Returns the solution's fitness (accuracy).

        Parameters
        ----------
        solution : list
            The network's types and subtypes

        Returns
        -------
        fitness : float
            The network's accuracy, in the range [0.0, 100.0].

        """
        descriptor = NeuralDescriptor()
        previous_in = self.data_channels
        for i in range(0, self.max_layers):
            # Get the layer's parameters
            dropout_rate = float(solution[i])
            # TODO: Implement weight scaling.
            weight_scale = float(solution[i+1*self.max_layers])

            filter_no = int(solution[i+2*self.max_layers])
            kernel_size = int(solution[i+3*self.max_layers])
            max_pool = True if int(
                solution[i+4*self.max_layers]) == 1 else False

            out_channels = filter_no

            # Define the layers and parameters
            conv_layer = nn.Conv2d
            conv_parameters = (previous_in, out_channels, kernel_size)

            previous_in = out_channels

            descriptor.add_layer_sequential(conv_layer, conv_parameters)

            dout = nn.Dropout
            dout_parameters = (dropout_rate, )
            descriptor.add_layer_sequential(dout, dout_parameters)
            if max_pool:
                pool = nn.MaxPool2d
                pool_parameters = (kernel_size, )
                descriptor.add_layer_sequential(pool, pool_parameters)

            descriptor.add_layer_sequential(nn.ReLU6, [])

        fitness = self.evaluator.descriptor_evaluate(descriptor,
                                                     untrained=self.untrained,
                                                     verbose=True) * 100

        return [fitness]
