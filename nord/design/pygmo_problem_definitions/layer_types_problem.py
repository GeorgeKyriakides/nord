"""
Created on 2018-08-02

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import warnings
import torch
import torch.nn as nn
from nord.neural_nets import NeuralDescriptor
from nord.neural_nets.layers import types_dict
from nord.neural_nets.distributed.distributed_configurator import DistributedConfig
from nord.neural_nets.distributed.environment import Environment
from nord.utils import generate_layer_parameters


class LayerTypeProblem:
    """A pygmo User Defined Problem, aimed at optimizing
    a Net's layers types.
    """

    def __init__(self, *, config: DistributedConfig = None, untrained=False, max_layers=2):
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
        self.config = config
        self.evaluator = None

        def get_types_and_max_no():
            """ Get the layers types nubmer
            and maximum number of sub-layer types
            """
            types_d = types_dict
            max_types = len(types_d)
            subtypes_no = []
            for key in types_d:
                subtypes_no.append(len(types_d[key]))

            max_subtypes = max(subtypes_no)
            return max_subtypes, max_types

        max_subtypes, max_types = get_types_and_max_no()
        low_bound = 0
        upper_bound_subtype = low_bound + max_subtypes
        upper_bound_type = low_bound + max_types

        for _ in range(max_layers):
            # Lower bound of layer's types
            self.bounds[0].append(low_bound)
            # Upper bound of layer's types
            self.bounds[1].append(upper_bound_type)
            # Lower bound of layer's subtypes
            self.bounds[0].append(low_bound)
            # Upper bound of layer's subtypes
            self.bounds[1].append(upper_bound_subtype)

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
            The network's types and subtypes

        Returns
        -------
        fitness : float
            The network's accuracy, in the range [0.0, 100.0].

        """

        # This needs to be here as pygmo pickles the object to send it to workers
        # and Environment contains thread locks which cannot be pickled
        if self.config is None:
            from nord.neural_nets.evaluators.neural_evaluators import LocalEvaluator
            warnings.warn('Empty config. Running locally')
            self.evaluator = LocalEvaluator(torch.optim.Adam, {}, True)

        if self.evaluator is None:
            self.evaluator = Environment(self.config)

        descriptor = NeuralDescriptor()
        previous_in = self.data_channels
        type_keys = list(types_dict.keys())
        for i in range(0, len(solution), 2):
            layer_type = int(solution[i])
            subtype = int(solution[i+1])

            layer_type = type_keys[layer_type]
            layer = types_dict[layer_type]
            subtype = subtype % len(layer)
            layer = layer[subtype]

            p_vals = generate_layer_parameters(layer)

            if 'in_channels' in p_vals:
                p_vals['in_channels'] = previous_in

            if 'out_channels' in p_vals:
                previous_in = p_vals['out_channels']

            descriptor.add_layer_sequential(layer, p_vals)
            descriptor.add_layer_sequential(nn.ReLU6, {})
        loss, metrics, time = self.evaluator.descriptor_evaluate(descriptor,
                                                                 epochs=1,
                                                                 untrained=self.untrained,
                                                                 dataset='cifar10')

        return [metrics['accuracy']*100]
