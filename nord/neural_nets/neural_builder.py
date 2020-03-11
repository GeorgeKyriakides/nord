"""
Created on 2018-08-05

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import torch.nn as nn
import torch
from .layers.layer_types import Flatten


class NeuralNet(nn.Module):
    """Basic class, implementing a deep neural net.
    """

    def __init__(self, net_descriptor, num_classes, sample, sort=False):
        """Generates the pytorch graph of the network.

        Parameters
        ----------
        net_descriptor : NeuralDescriptor
            A descriptor of the netowrk's structure.
            (see :class:`~.neural_descriptor.NeuralDescriptor`)

        num_classes : int
            The number of classes (output layer neurons) of the network.

        sample : Tensor
            A single example of the train/test set.

        sort : bool
            If set to True, the depth of each layer will be
            added before its name.


        """
        super(NeuralNet, self).__init__()

        self.this_connections = net_descriptor.connections

        self.first_layer = net_descriptor.first_layer
        self.last_layer = net_descriptor.last_layer

        self.this_layers = dict()
        self.this_num_classes = num_classes
        layer_dict = self.__name_all_paths(net_descriptor.layers)
        for key in layer_dict.keys():
            layer, params = layer_dict[key]
            layer_instance = layer(*params)
            setattr(self, key, layer_instance)
            self.this_layers[key] = getattr(self, key)
        self.flatten = Flatten()
        in_sz = self.__get_last_layer_size(sample)
        self.final_layer = nn.Linear(in_sz, self.this_num_classes)

    def __get_last_layer_size(self, instance_sample):
        """Returns the input size for the last (classification) layer,
        by running a simple forward pass through all the network's layers.

        Parameters
        ----------
        instance_sample : int
            A single example of the train/test set.


        Returns
        ----------
        size : int
            The input size for the last (classification) layer.
        """
        with torch.no_grad():
            x = self.this_layers[self.first_layer](instance_sample)
            outs = {self.first_layer: x}
            keys = self.this_connections.keys()
            # Example: key='layer_0'
            for key in sorted(keys):
                # to_layer = 'layer_1_1'
                for to_layer in self.this_connections[key]:
                    # tmp_out = layer_1_1(x)
                    tmp_out = self.this_layers[to_layer](outs[key])
                    # outs += {'layer_1_1': tmp_out}
                    outs[to_layer] = tmp_out

            last = outs[self.last_layer]
        return last.view(last.size()[0], -1).size()[1]

    def forward(self, x):
        """ Implement the forward pass
        """

        x = self.this_layers[self.first_layer](x)
        outs = {self.first_layer: x}
        keys = self.this_connections.keys()
        # Example: key='layer_0'
        for key in sorted(keys):
            # to_layer = 'layer_1_1'
            for to_layer in self.this_connections[key]:
                # tmp_out = layer_1_1(x)
                tmp_out = self.this_layers[to_layer](outs[key])
                # outs += {'layer_1_1': tmp_out}
                outs[to_layer] = tmp_out

        flats = self.flatten(outs[self.last_layer])
        x = self.final_layer(flats)
        return x

    def __name_all_paths(self, layers_in):
        """Find all the possible paths from the input layer to the output layer
           and name the nodes according to their depth.
        """

        layers = layers_in

        def find_all_paths(graph, start, end, path=[]):
            """From python.org
            """
            path = path + [start]
            if start == end:
                return [path]
            if start not in graph:
                return []
            paths = []

            for node in graph[start]:
                if node not in path:
                    newpaths = find_all_paths(graph, node, end, path)
                    for newpath in newpaths:
                        paths.append(newpath)

            return paths

        def set_layer_level(layer, level, renamed):
            new_key = str(level)+'_'+layer
            layers[new_key] = layers.pop(layer)
            self.this_connections[new_key] = self.this_connections.pop(layer)

            if layer == self.first_layer:
                self.first_layer = new_key
            elif layer == self.last_layer:
                self.last_layer = new_key
            return new_key

        graph = self.this_connections
        start = self.first_layer
        end = self.last_layer
        paths = find_all_paths(graph, start, end)
        renamed = {}
        for path in paths:
            level = 0
            for node in path:
                if node not in renamed:
                    new_node = set_layer_level(node, level, renamed)
                    renamed[node] = new_node
                level += 1

        for new_key in self.this_connections:
            for i in range(len(self.this_connections[new_key])):
                node = self.this_connections[new_key][i]
                if node in renamed:
                    self.this_connections[new_key][i] = renamed[node]
        return layers
