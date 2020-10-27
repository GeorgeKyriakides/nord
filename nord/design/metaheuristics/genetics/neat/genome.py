"""
Created on 2018-10-29

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import copy

import networkx as nx
import torch.nn as nn
import numpy as np


from nord.neural_nets import NeuralDescriptor
from nord.neural_nets.layers import Identity, ScaleLayer

from nord.utils import get_random_value

from .chromosome import Chromosome
from .gene import ConnectionGene, LayerGene

INPUT = -2
OUTPUT = -1


class Genome(object):

    def __init__(self, layer_bounds_types, layer_bounds,
                 add_node_rate, add_connection_rate,
                 mutation_rate, innovation=None):

        self.connections = Chromosome()
        self.nodes = Chromosome()
        self.layer_bounds = layer_bounds
        self.layer_bounds_types = layer_bounds_types
        self.add_node_rate = add_node_rate
        self.add_connection_rate = add_connection_rate
        self.mutation_rate = mutation_rate

        # Add initial structure
        start_node = LayerGene(None, None, io_node=True)
        end_node = LayerGene(None, None, io_node=True)
        start_node.innovation_number = INPUT
        end_node.innovation_number = OUTPUT
        self.nodes.add_gene(start_node)
        self.nodes.add_gene(end_node)

        connection_node = ConnectionGene(
            start_node.innovation_number, end_node.innovation_number)
        if innovation is not None:
            innovation.assign_number(connection_node)
        self.connections.add_gene(connection_node)
        self.innovation = innovation

    def mutate(self):
        r = get_random_value()

        if r < self.add_node_rate:
            g = np.random.choice(list(self.connections.genes.keys()))
            gene = self.connections.genes[g]

            start_node, end_node = gene.value
            gene.enabled = False

            new_node = LayerGene(self.layer_bounds_types, self.layer_bounds)
            self.innovation.assign_number(new_node)

            new_start = ConnectionGene(
                start_node, new_node.innovation_number)
            self.innovation.assign_number(new_start)

            new_end = ConnectionGene(
                new_node.innovation_number, end_node)
            self.innovation.assign_number(new_end)

            self.connections.add_gene(new_start)
            self.connections.add_gene(new_end)
            self.nodes.add_gene(new_node)

        elif r < self.add_node_rate + self.add_connection_rate:
            nodes = list(self.nodes.genes)
            nodes.remove(INPUT)
            nodes.remove(OUTPUT)
            start_node = np.random.choice(nodes+[INPUT])
            end_node = np.random.choice(nodes+[OUTPUT])
            new_node = ConnectionGene(start_node, end_node)

            if new_node not in self.connections.genes.values():

                self.innovation.assign_number(new_node)
                self.connections.add_gene(new_node)

            self.connections.mutate(self.mutation_rate)
            self.nodes.mutate(self.mutation_rate)

    def crossover(self, other):
        new = copy.deepcopy(self)

        new.connections = new.connections.crossover(other.connections)
        new.nodes = new.nodes.crossover(other.nodes)
        return new

    def __repr__(self):
        return str({'Connections': self.connections, 'Nodes': self.nodes})

    def __hash__(self):
        return hash(self.connections) + hash(self.nodes)

    def __eq__(self, other):
        """Overrides the default implementation"""

        if isinstance(other, Genome):

            if not (len(self.nodes.genes) == len(other.nodes.genes) and
                    len(self.connections.genes) == len(other.connections.genes)):
                return False

            for i in self.nodes.genes:
                if not self.nodes.genes[i] == other.nodes.genes[i]:
                    return False

            for i in self.connections.genes:
                if not self.connections.genes[i] == other.connections.genes[i]:
                    return False

        return True

    @staticmethod
    def __from_repr__(rpr):
        import ast
        g = Genome([int, float, float, int, bool],
                   [[32, 0.0, 0, 1, 0],
                    [256, 0.7, 2.0, 3, 1]], [], [],
                   0.1, 0.1, 0.1, None)
        rpr = ast.literal_eval(rpr)
        connections = rpr['Connections']
        for innovation in connections:
            g.connections.genes[innovation] = ConnectionGene.__from_repr__(
                connections[innovation])
            g.connections.index.append(innovation)

        g.connections.genes.pop(None)

        nodes = rpr['Nodes']
        for innovation in nodes:
            g.nodes.genes[innovation] = LayerGene.__from_repr__(
                nodes[innovation])
            g.nodes.index.append(innovation)

        return g

    def to_descriptor(self, dimensions=2):

        self.active_nodes = 0
        descriptor = NeuralDescriptor()

        actives = set()
        self.actives = set()
        # Get only active nodes
        for p in self.get_direct_paths():
            for n in p:
                actives.add(n)

        # First add the nodes themselves
        for g in self.nodes.genes:
            gene = self.nodes.genes[g]
            # Don't add inactive nodes
            if gene.innovation_number in actives and gene.enabled:
                if not gene.io:
                    self.active_nodes += 1
                    self.actives.add(str(gene.value))
                    # Get the node's name (innovation number)
                    innv = str(gene.innovation_number)

                    # Get the parameters
                    parameters = gene.value

                    filter_no = int(parameters[0])
                    dropout_rate = float(parameters[1])
                    weight_scale = float(parameters[2])
                    kernel_size = int(parameters[3])

                    max_pool = True if int(parameters[4]) == 1 else False
                    out_channels = filter_no

                    # --Define the layers and parameters--

                    # Convolution layer
                    conv_layer = nn.Conv2d
                    if dimensions == 1:
                        conv_layer = nn.Conv1d
                    conv_parameters = {'in_channels': 1000,
                                       'out_channels': out_channels,
                                       'kernel_size': kernel_size}

                    descriptor.add_layer(
                        conv_layer, conv_parameters, name=innv+'in')

                    # Scale the weights
                    descriptor.add_layer_sequential(
                        ScaleLayer, {'scale': weight_scale}, name=innv+'scale')

                    # Dropout layer
                    if dimensions == 2:
                        dout = nn.Dropout2d
                    else:
                        dout = nn.Dropout
                    dout_parameters = {'p': dropout_rate}
                    descriptor.add_layer_sequential(
                        dout, dout_parameters, name=innv+'dout')

                    # Max pool layer
                    if max_pool:
                        pool = nn.MaxPool2d
                        if dimensions == 1:
                            pool = nn.MaxPool1d
                        pool_parameters = {'kernel_size': kernel_size,
                                           'stride': kernel_size}
                        descriptor.add_layer_sequential(
                            pool, pool_parameters, name=innv+'pool')

                    # Activation layer
                    descriptor.add_layer_sequential(
                        nn.ReLU6, {}, name=innv+'out')

        # Add IO layers
        descriptor.add_layer(Identity, {}, name='-2out')
        descriptor.add_layer(Identity, {}, name='-1in')
        descriptor.first_layer = '-2out'
        descriptor.last_layer = '-1in'

        # Connect the layers
        for g in self.connections.genes:
            gene = self.connections.genes[g]
            from_, to_ = gene.value
            # Connect all active
            if gene.enabled:
                # Only connecitons from/to active nodes should be added
                if from_ in actives and to_ in actives:

                    last_out = str(from_)+'out'
                    descriptor.connect_layers(last_out, str(to_)+'in')

        return descriptor

    def plot(self):
        import matplotlib.pyplot as plt

        def my_layout(G, paths):
            nodes = G.nodes
            lengths = [-len(x) for x in paths]
            sorted_ = np.argsort(lengths)

            positions = dict()
            h = 0
            w = 0

            for index in sorted_:
                h = 0
                added = False
                path = paths[index]
                for node in path:
                    if node not in positions:
                        positions[node] = (w, h)
                        added = True
                        h -= 1
                    else:
                        if h > positions[node][1]:
                            h = positions[node][1]

                if added:
                    if w >= 0:
                        w += 1
                    w *= -1

            h = 0
            for node in nodes:
                if node not in positions:
                    positions[node] = (w, h)
                    h -= 1

            if -1 in positions:
                positions[-1] = (positions[-1][0], positions[-1][1]-1)
            if -2 in positions:
                positions[-2] = (positions[-2][0], positions[-2][1]+1)
            return positions

        G = self.to_networkx()
        plt.figure()
        in_path = self.get_direct_paths()
        # pos = graphviz_layout(G, root='-2')
        pos = my_layout(G, in_path)
        nx.draw(G, pos=pos, node_color='b', with_labels=True)

        nodes = set()
        for p in in_path:
            for node in p:
                nodes.add(node)
        nx.draw_networkx_nodes(G, pos=pos,
                               node_color='r',
                               nodelist=list(nodes),
                               with_labels=True)
        plt.show()

    def to_networkx(self, active_only=True):

        G = nx.DiGraph()
        for g in self.connections.genes:
            gene = self.connections.genes[g]
            if gene.enabled or not active_only:
                G.add_edge(*gene.value)

        return G

    def get_direct_paths(self):
        G = self.to_networkx()
        try:
            paths = nx.all_simple_paths(G, INPUT, OUTPUT)
        except nx.NodeNotFound:
            paths = [[]]
        return [p for p in paths]

    def get_recursions(self):
        G = self.to_networkx()
        cycles = nx.simple_cycles(G)
        return [c for c in cycles]

    def get_incoming_layers(self):
        G = self.to_networkx()
        incoming = dict()
        edges = G.edges()
        for edge in edges:
            from_ = edge[0]
            to_ = edge[1]
            if to_ in incoming:
                incoming[to_].append(from_)
            else:
                incoming[to_] = [from_]
        return incoming

    def get_connection_ratio(self):
        G = self.to_networkx()
        p = len(self.get_direct_paths())
        r = (p**2)/G.number_of_nodes()
        return r

    def remove_recursions(self):
        recs = self.get_recursions()
        recs.sort(key=len)

        edges = set()
        for rec in recs:

            if len(rec) == 1:
                start = rec[0]
                end = rec[0]
                edges.add((start, end))
            else:
                for i in range(1, len(rec)):
                    start = rec[i-1]
                    end = rec[i]
                    edges.add((start, end))

        for c in self.connections.genes:
            if self.connections.genes[c].value in edges:
                self.connections.genes[c].enabled = False
