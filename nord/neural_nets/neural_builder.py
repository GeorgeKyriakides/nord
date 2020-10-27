
"""
Created on Sat Aug  4 18:24:35 2018

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import copy
from typing import Dict, List, Tuple, Type

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nord.configurations.dense_parts import get_dense_net
from nord.utils import get_layer_out_size, get_transpose_out_size
from torch import Tensor
from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin
from torch.nn.modules.pooling import _AvgPoolNd, _MaxPoolNd

from .layers import Conv2d151, Flatten, Identity, SizeScaleLayer, types_dict
from .neural_descriptor import NeuralDescriptor

MIN_MAX_DIMENSIONS = 'MIN'


class NeuralNet(nn.Module):
    """Basic class, implementing a deep neural net.
    """

    def __init__(self, net_descriptor: NeuralDescriptor, num_classes: int,
                 input_shape: Tuple,
                 input_channels: int, sort: bool = False,
                 untrained: bool = False,
                 keep_dimensions: int = None,
                 dense_part=None):
        """Generates the pytorch graph of the network.

        Parameters
        ----------
        net_descriptor : NeuralDescriptor
            A descriptor of the netowrk's structure.
            (see :class:`~.neural_descriptor.NeuralDescriptor`)

        num_classes : int
            The number of classes (output layer neurons) of the network.

        input_shape : tuple
            Shape of inputs.

        input_channels : int
            Number of input channels.

        sort : bool
            If set to True, the depth of each layer will be
            added before its name.

        keep_dimensions: int
            This parameter dicatates the rescaling of layer outputs.
            The outputs are rescaled to 'keep_dimensions' size.
            If None, the network will not rescale the outputs at all.

        """
        super().__init__()

        self.descriptor = copy.deepcopy(net_descriptor)
        self.connections = self.descriptor.connections
        self.in_connections = self.descriptor.incoming_connections
        self.input_shapes = dict()
        self.input_channels = dict()

        self.first_layer = self.descriptor.first_layer
        self.last_layer = self.descriptor.last_layer
        self.keep_dimensions = keep_dimensions

        self.functional = True
        if len(self.in_connections[self.last_layer]) == 0:
            self.functional = False
            return

        if len(self.connections[self.first_layer]) == 0:
            self.functional = False
            return

        self.layers = dict()
        self.num_classes = num_classes

        # Get all paths
        layer_dict = self.__name_all_paths(self.descriptor.layers)

        self.input_shapes[self.first_layer] = input_shape
        self.input_channels[self.first_layer] = input_channels
        self.data_shape = len(input_shape)

        self.hidden_outs = dict()
        self.recursions = [[x[-1], x[0]] for x in self.get_recursions()]
        self.recursion_paths = [x[1:-1] for x in self.get_recursions()]
        self.recursion_origins = [x[0] for x in self.recursions]

        self.scale_all_layers()
        self.connections = self.descriptor.connections
        self.in_connections = self.descriptor.incoming_connections
        # Instantiate all layers
        for key in layer_dict.keys():
            # Get layer type and parameters
            layer, params = layer_dict[key]
            # Get instance
            layer_instance = layer(**params)
            if untrained:
                for param in layer_instance.parameters():
                    param.requires_grad = False
            # Set as attribute
            setattr(self, key, layer_instance)
            self.layers[key] = getattr(self, key)

        self.flatten = Flatten()

        in_sz = self.__get_flat_out_size_dynamic()
        
        if dense_part is None:
            dense_part = get_dense_net(1, num_classes, nn.Identity())

        fc_layers = dense_part(in_sz)

        self.fc_layers = {}
        for key, value in fc_layers.items():
            setattr(self, key, value)
            self.fc_layers[key] = getattr(self, key)

    def __get_flat_out_size_dynamic(self):
        self.eval()  # Set to eval so batchnorm won't complain
        input_channels = self.input_channels[self.first_layer]
        input_shape = self.input_shapes[self.first_layer]
        dummy = torch.randn([1, input_channels, *input_shape])
        out = self.flatten(self.__internal_fwd(dummy))
        final_sz = list(out.shape)[-1]

        self.train()  # ReSet to train
        return final_sz

    def __internal_fwd(self, x: Tensor):
        """ Implement the forward pass
        """
        x = self.layers[self.first_layer](x)
        outs = {self.first_layer: x}

        keys = self.connections.keys()
        # Example: key='layer_1_1'
        for key in sorted(keys):
            if key is self.first_layer:
                continue
            other_in = None
            inputs = self.in_connections[key]
            for from_layer in set(inputs):

                if int(from_layer.split('_')[0]) > int(key.split('_')[0]):
                    continue

                else:
                    tmp_in = outs[from_layer]

                other_in = 0 if other_in is None else other_in
                if other_in is not None:
                    other_in = tmp_in + other_in

                else:
                    other_in = tmp_in

            outs[key] = self.layers[key](other_in)

        return outs[self.last_layer]

    def forward(self, x: Tensor):

        x = self.flatten(self.__internal_fwd(x))
        for layer in self.fc_layers:
            x = self.fc_layers[layer](x)

        return x

    def scale_all_layers(self):
        """ Scale all inputs for multi-input layers.
        """
        self.eval()  # Evaluation mode so batchnorm won't complain
        scaled = set()
        scaled.add(self.first_layer)
        not_scaled = list(self.descriptor.layers.keys())
        not_scaled.remove(self.first_layer)
        self.pure_recursions = set()
        while len(not_scaled) > 0:

            for target in not_scaled:

                origins = []
                tmp = self.in_connections[target]
                origin_set = set(tmp)

                # If the origins are already scaled, proceed
                if origin_set <= scaled:
                    self.scale_target_inputs(target, origin_set)
                    scaled.add(target)
                    not_scaled.remove(target)

        self.eval()  # Back to training mode

    def scale_target_inputs(self, target: str, origins: List[str]):
        """ Scale the selected origins layers, in order to concatenate
            inputs to target.

            Parameters
            ----------
            target : string
                Name of the input layer

            origins : list[string]
                Names of output layers
        """

        min_sz, min_channels = self.get_min_input_size(target, origins)
        if self.keep_dimensions is not None:
            min_sz = self.keep_dimensions

        if 'in_channels' in self.descriptor.layers[target][1]:
            self.descriptor.layers[target][1]['in_channels'] = min_channels
        elif 'num_features' in self.descriptor.layers[target][1]:
            self.descriptor.layers[target][1]['num_features'] = min_channels

        self.input_shapes[target] = min_sz
        self.input_channels[target] = min_channels

        for origin in origins:
            self.scale_layer_outputs(origin, target, min_sz, min_channels)

    def scale_layer_outputs(self, origin: str, target: str, min_sz: int, min_channels: int):
        """ Scale the selected origin layer, in order to concatenate
            inputs to target.

            Parameters
            ----------
            target : string
                Name of the input layer

            origin : string
                Names of output layer

            min_sz : int
                The desired output size
                (minimum size that enables concatenation)

            min_channels : int
                The desired output channels
                (minimum channels that enable concatenation)
        """

        shape = self.get_output_shape(origin)
        channels = shape[1]
        dimensions = shape[2:]

        origin_out = dimensions
        out_ch = channels

        layer_name = origin+'_0sizescale_'+target
        ch_name = origin+'_1chanscale_'+target

        layer = None
        params = None
        # If the target is a flatten layer do nothing.
        if (self.descriptor.layers[target][0] is Flatten):
            return
        if (self.descriptor.layers[origin][0] is nn.Linear) and (self.descriptor.layers[target][0] is nn.Linear):
            return
        # Else if the output size differs from the input size
        # scale the output size
        elif not origin_out == min_sz:
            layer = SizeScaleLayer
            params = {'final_size': min_sz}

        if layer is not None:
            self.descriptor.add_layer(layer, params, name=layer_name)
            self.descriptor.connect_layers(origin, layer_name)
            self.descriptor.connect_layers(layer_name, target)
            self.descriptor.disconnect_layers(origin, target)

            self.input_shapes[layer_name] = origin_out
            self.input_channels[layer_name] = out_ch
            self.input_shapes[target] = min_sz

        # Scale channels as well
        if out_ch is not min_channels:
            ch_layer = nn.Conv2d
            if self.data_shape == 1:
                ch_layer = nn.Conv1d

            ch_params = {'in_channels': out_ch, 'out_channels': min_channels,
                         'kernel_size': 1}
            self.descriptor.add_layer(ch_layer, ch_params, name=ch_name)
            if layer is not None:
                self.descriptor.connect_layers(layer_name, ch_name)
                self.descriptor.connect_layers(ch_name, target)
                self.descriptor.disconnect_layers(layer_name, target)

            else:
                self.descriptor.connect_layers(origin, ch_name)
                self.descriptor.connect_layers(ch_name, target)
                self.descriptor.disconnect_layers(origin, target)

            self.input_shapes[ch_name] = self.input_shapes[target]
            self.input_channels[ch_name] = self.input_channels[target]
            self.input_channels[target] = min_channels

    def get_min_input_size(self, target: str, origins: List[str]):
        """ Get the minimum input size that enables concatenation.

            Parameters
            ----------
            target : string
                Name of the input layer

            origins : list[string]
                Names of output layers

            Returns
            -------
            min_dimensions : int
                Minimum size

            min_channels : int
                Minimum channels

        """
        if self.descriptor.layers[target][0] is nn.Linear:
            return self.descriptor.layers[target][1]['in_features'], 1

        min_channels = 0
        min_dimensions = np.zeros(len(self.input_shapes[self.first_layer]))

        if MIN_MAX_DIMENSIONS == 'MIN':
            min_channels += 1e+10
            min_dimensions += 1e+10

        for node in origins:
            shape = self.get_output_shape(node)
            channels = shape[1]
            dimensions = shape[2:]

            if MIN_MAX_DIMENSIONS == 'MAX':
                min_dimensions = np.maximum(dimensions, min_dimensions)
                min_channels = np.maximum(channels, min_channels)
            else:
                min_dimensions = np.minimum(dimensions, min_dimensions)
                min_channels = np.minimum(channels, min_channels)

        # Assure input is not smaller than kernel size
        if 'kernel_size' in self.descriptor.layers[target][1]:
            kernel = self.descriptor.layers[target][1]['kernel_size']
            if (min_dimensions < kernel).any():
                min_dimensions = np.zeros(
                    len(self.input_shapes[self.first_layer]))+kernel

        return min_dimensions.astype(int).tolist(), min_channels.astype(int).tolist()

    def get_output_shape(self, node: str):
        """Calculate the output shape of a layer

        Parameters
        ----------
        node : string
            Name of the layer

        Returns
        -------
        The layer's output shape

        """
        node_class = self.descriptor.layers[node][0]
        params = self.descriptor.layers[node][1]

        input_shapes = self.input_shapes[node]
        input_channels = self.input_channels[node]

        dummy = torch.randn((1, input_channels, *input_shapes))
        dummy_layer = node_class(**params)
        dummy_out = dummy_layer(dummy)
        return list(dummy_out.size())

    def remove_connection(self, origin: str, target: str):
        """Remove a connection from the network

        Parameters
        ----------
        origin : string
            Name of the origin layer

        target : string
            Name of the target layer

        """

        while target in self.connections[origin]:
            self.connections[origin].remove(target)
        while origin in self.in_connections[target]:
            self.in_connections[target].remove(origin)

    def add_connection(self, origin: str, target: str):
        """Add a connection to the network

        Parameters
        ----------
        origin : string
            Name of the origin layer

        target : string
            Name of the target layer

        """
        if origin not in self.connections:
            self.connections[origin] = []
            self.in_connections[origin] = []
        if target not in self.connections:
            self.connections[target] = []
            self.in_connections[target] = []

        self.in_connections[target].append(origin)
        self.connections[origin].append(target)

    def __name_all_paths(self, layers_in: Dict[str, List],
                         acyclic: bool = True):
        """Find all the possible paths from the input layer to the output layer
           and name the nodes according to their depth.

        Parameters
        ----------
        layers_in : dict
            Contains the layer name as key. The dictionary's items
            are the layer's type and its parameters.

        acyclic : bool
            If true, only acyclic graphs are considered.

        Returns
        ----------

        A dictionary with updated names
        """

        layers = layers_in

        def set_layer_level(layer, level):
            new_key = str(level).zfill(3)+'_'+layer
            layers[new_key] = layers.pop(layer)
            self.connections[new_key] = self.connections.pop(layer)
            self.in_connections[new_key] = self.in_connections.pop(layer)

            if layer == self.first_layer:
                self.first_layer = new_key
            elif layer == self.last_layer:
                self.last_layer = new_key
            return new_key

        paths = None
        if acyclic:
            paths = self.get_acyclic_paths()
        else:
            paths = self.get_direct_paths()

        renamed = {}
        paths = sorted(paths, reverse=True, key=len)
        for path in paths:
            level = 1
            for node in path:
                if node not in renamed:
                    if node == self.last_layer:
                        level += 1
                    if node == self.first_layer:
                        level -= 1
                    new_node = set_layer_level(node, level)
                    renamed[node] = new_node

                else:
                    level = int(renamed[node].split('_')[0])
                level += 1

        for new_key in self.connections:
            for i in range(len(self.connections[new_key])):
                node = self.connections[new_key][i]
                if node in renamed:
                    self.connections[new_key][i] = renamed[node]

        for new_key in self.in_connections:
            for i in range(len(self.in_connections[new_key])):
                node = self.in_connections[new_key][i]
                if node in renamed:
                    self.in_connections[new_key][i] = renamed[node]
        return layers

    def to_networkx(self):
        """Create a networkx graph from the network

        Returns
        -------
        G : MultiDiGraph

        """

        G = nx.MultiDiGraph()
        for start in self.connections:
            ends = self.connections[start]
            for end in ends:
                G.add_edge(start, end)

        return G

    def get_direct_paths(self):
        """Return all the direct paths from the input to the output layer

        Returns
        -------
        list with all the direct paths
        """
        G = self.to_networkx()
        try:
            paths = nx.all_simple_paths(G, self.first_layer, self.last_layer)
        except nx.NodeNotFound:
            paths = [[]]
        return [p for p in paths]

    def get_acyclic_paths(self):
        """Return all the direct acyclic paths from the input to the output layer

        Returns
        -------
        list with all the direct paths
        """
        G = self.to_networkx()
        # recs = self.get_recursions()

        try:
            paths = nx.all_simple_paths(G, self.first_layer, self.last_layer)
        except nx.NodeNotFound:
            paths = [[]]

        rets = []
        for p in paths:
            if p not in rets:
                rets.append(p)
        return rets

    def get_recursions(self):
        """Return all the recursions in the network

        Returns
        -------
        list with all the recursions
        """
        G = self.to_networkx()
        cycles = nx.simple_cycles(G)
        return [sorted(c) for c in cycles]

    def plot(self, title: str = None):
        import matplotlib.pyplot as plt
        import numpy as np

        spacing_h = 3.0
        spacing_w = 0.5
        half_spacing = 0.25

        def my_layout(G, paths, recursions):
            nodes = G.nodes
            lengths = [-len(x) for x in paths]
            sorted_ = np.argsort(lengths)

            positions = dict()
            h = 0
            w = 0
            min_x, max_x = -spacing_w, spacing_w

            for index in sorted_:
                h = 0
                added = False
                path = paths[index]
                for node in path:
                    if node not in positions:
                        positions[node] = (w, h)
                        added = True
                        h -= spacing_h
                    else:
                        if h > positions[node][1]:
                            h = positions[node][1]

                if added:
                    if w >= 0:
                        w += spacing_w
                    w *= -1
                    if w > max_x:
                        max_x = w
                    if w < min_x:
                        min_x = w

            h = 0
            for node in nodes:
                if node not in positions:
                    positions[node] = (w, h)
                    h -= spacing_h

            f_l = self.first_layer
            l_l = self.last_layer
            if f_l in positions:
                positions[f_l] = (positions[f_l][0],
                                  positions[f_l][1]+spacing_h)
            if l_l in positions:
                positions[l_l] = (positions[l_l][0],
                                  positions[l_l][1]-spacing_h)

            recursed_nodes = []
            for path in recursions:
                last = sorted(path)[-1]
                if last not in recursed_nodes:
                    positions[last] = (positions[last][0]+half_spacing,
                                       positions[last][1])
                    recursed_nodes.append(last)
            return positions, min_x, max_x

        G = self.to_networkx()
        plt.figure()
        plt.title(title)
        ax = plt.gca()
        in_path = self.get_direct_paths()
        recs = self.get_recursions()
        pos, min_x, max_x = my_layout(G, in_path, recs)

        nodes = set()
        for p in in_path:
            for node in p:
                nodes.add(node)
        for p in recs:
            for node in p:
                nodes.add(node)

        labels = {}

        for n in nodes:
            wrap_chars = 15
            name = str(self.layers[n])
            labels[n] = '\n'.join(name[i:i+wrap_chars]
                                  for i in range(0, len(name), wrap_chars))

        nx.draw(G, pos=pos,
                with_labels=True,
                node_shape="s",
                node_color="none",
                bbox=dict(facecolor="skyblue", edgecolor='black',
                          boxstyle='round,pad=0.2', alpha=0.5),
                labels=labels,
                font_size=8,
                nodelist=list(nodes))

        ax.set_xlim(xmin=min_x-half_spacing, xmax=max_x+half_spacing)
        plt.show()
