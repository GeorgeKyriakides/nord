"""
Created on 2018-10-29

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

import networkx as nx
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import AvgPooling, MaxPooling, SAGEConv, SortPooling

from nord.neural_nets import NeuralDescriptor
from nord.neural_nets.layers import Identity
from nord.utils import get_random_value

from .chromosome import Chromosome
from .gene import ConnectionGene, LayerGene, HyperparameterGene
from .rem_config import (DROPOUT_PROB, FULL_CHANNELS_NO, INPUT, INPUT_NAME,
                         OUTPUT, OUTPUT_NAME, UNEVALUATED_FITNESS, POOLING,
                         POOLING_NAME)

# # NASBENCH IJNS paper
# layers_list = ['SAGE_LSTM.H', 'SAGE_LSTM.N', 'SAGE_LSTM.D',
#                'SAGE_GCN.H', 'SAGE_GCN.N', 'SAGE_GCN.D']

# FMNIST NCA paper
layers_list = ['SAGE_LSTM.H', 'SAGE_LSTM.N', 'SAGE_LSTM.D',
               'SAGE_POOL.H', 'SAGE_POOL.N', 'SAGE_POOL.D']


pooling_list = ['POOL_1.S', 'POOL_2.S', 'POOL_3.S', 'POOL_1.A', 'POOL_1.M']


class Genome(object):

    def __init__(self, identity_rate, add_node_rate, channels, strides,
                 innovation=None):

        self.connections = Chromosome()
        self.nodes = Chromosome()
        self.hyperparameters = Chromosome()
        self.layer_bounds = [[0], [len(layers_list)]]
        self.hyperparameter_bounds = [[0], [len(pooling_list)]]
        self.layer_bounds_types = [int]
        self.hyperparameter_bounds_types = [int]
        self.identity_rate = identity_rate
        self.add_node_rate = add_node_rate
        self.channels = channels
        self.strides = strides

        self.fitness = UNEVALUATED_FITNESS

        # Add initial structure
        start_node = LayerGene(None, None, io_node=True)
        end_node = LayerGene(None, None, io_node=True)
        start_node.innovation_number = INPUT
        end_node.innovation_number = OUTPUT
        self.nodes.add_gene(start_node)
        self.nodes.add_gene(end_node)

        pooling_gene = HyperparameterGene(
            self.hyperparameter_bounds_types, self.hyperparameter_bounds)
        self.hyperparameters.add_gene(pooling_gene)

        connection_node = ConnectionGene(
            start_node.innovation_number, end_node.innovation_number)
        if innovation is not None:
            innovation.assign_number(connection_node)
        self.connections.add_gene(connection_node)
        self.innovation = innovation

    def mutate(self, identity_rate=None, add_node_rate=None):

        if identity_rate is None:
            identity_rate = self.identity_rate

        if add_node_rate is None:
            add_node_rate = self.add_node_rate

        r = get_random_value()
        self.fitness = UNEVALUATED_FITNESS

        if r < add_node_rate:
            # Add a node
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

        elif r > identity_rate + add_node_rate:
            # Mutate a node, a connection, or the pooling
            r_t = get_random_value()

            if r_t < 0.33:  # Mutate a node
                g = np.random.choice(list(self.nodes.genes.keys()))
                gene = self.nodes.genes[g]
                gene.mutate(1.0)

            elif r_t < 0.66:  # Mutate a connection
                g = np.random.choice(list(self.connections.genes.keys()))
                gene = self.connections.genes[g]
                gene.mutate(1.0)
                # start_node, end_node = gene.value

                # nodes = list(self.nodes.genes)
                # nodes.remove(INPUT)
                # nodes.remove(OUTPUT)

                # end_node = np.random.choice(nodes+[OUTPUT])
                # gene.value = (start_node, end_node)
            else:
                g = np.random.choice(list(self.hyperparameters.genes.keys()))
                gene = self.hyperparameters.genes[g]
                gene.mutate(1.0)

    def __repr__(self):
        return str({'Connections': self.connections, 'Nodes': self.nodes, 'Hypers': self.hyperparameters})

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

            for i in self.hyperparameters.genes:
                if not self.hyperparameters.genes[i] == other.hyperparameters.genes[i]:
                    return False

        return True

    @staticmethod
    def __from_repr__(rpr):
        import ast
        g = Genome(0.05, 0.2, FULL_CHANNELS_NO, 1)
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
                nodes[innovation], g.layer_bounds_types, g.layer_bounds)
            g.nodes.index.append(innovation)

        hyperparameters = rpr['Hypers']
        for innovation in hyperparameters:
            g.hyperparameters.genes[innovation] = LayerGene.__from_repr__(
                hyperparameters[innovation], g.hyperparameter_bounds_types, g.hyperparameter_bounds)
            g.hyperparameters.index.append(innovation)

        return g

    def to_descriptor(self, dimensions=1):

        assert dimensions == 1

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
                    selected_layer, params = layers_list[gene.value[0]], None
                    if '.' in selected_layer:
                        selected_layer, params = selected_layer.split('.')

                    layer, aggregator = selected_layer.split('_')
                    channels = self.channels
                    aggregator = aggregator.lower()
                    sage = False
                    if layer == 'SAGE':
                        # Not used for now
                        sage = True
                        if params == 'H':
                            channels = int(channels/2)
                        elif params == 'D':
                            channels = int(channels*2)
                        layer = SAGEConv
                        parameters = {'in_feats': 1000,
                                      'out_feats': channels,
                                      'aggregator_type': aggregator,
                                      'feat_drop': DROPOUT_PROB,
                                      'activation': F.relu}

                    descriptor.add_layer(layer, parameters, name=innv+'in')
                    # Activation layer
                    # if conv:
                    #     descriptor.add_layer_sequential(
                    #         nn.ReLU6, {}, name=innv+'relu')
                    #     descriptor.add_layer_sequential(
                    #         nn.BatchNorm2d, {'num_features': channels},
                    #         name=innv+'batchnorm')
                    #     descriptor.add_layer_sequential(
                    #         nn.Dropout2d, {'p': DROPOUT_PROB}, name=innv+'dropout')

                    descriptor.add_layer_sequential(
                        Identity, {}, name=innv+'out')

        # Add IO layers
        descriptor.add_layer(Identity, {}, name=INPUT_NAME)
        descriptor.add_layer(Identity, {}, name=OUTPUT_NAME)
        descriptor.first_layer = INPUT_NAME
        descriptor.last_layer = OUTPUT_NAME

        pooling_gene = self.hyperparameters.genes[list(self.hyperparameters.genes.keys())[
            0]]
        pooling_layer, params = pooling_list[pooling_gene.value[0]].split(
            '.')
        pooling_layer, size = pooling_layer.split('_')

        layer = None
        parameters = {}
        if params == 'A':
            layer = AvgPooling
        if params == 'M':
            layer = MaxPooling
        else:
            layer = SortPooling
            parameters = {'k': int(size)}

        descriptor.add_layer(layer, parameters, name=POOLING_NAME)

        # Connect the layers
        for g in self.connections.genes:
            gene = self.connections.genes[g]
            from_, to_ = gene.value
            # Connect all active
            if gene.enabled:
                # Only connecitons from/to active nodes should be added
                if from_ in actives and to_ in actives:

                    last_out = str(from_)+'out'
                    to_layer = str(to_)+'in'

                    if from_ == INPUT:
                        last_out = INPUT_NAME
                    # This is different in order to add pooling as "last" layer
                    elif from_ == OUTPUT:
                        last_out = POOLING_NAME

                    if to_ == INPUT:
                        to_layer = INPUT_NAME
                    # This is different in order to add pooling as "last" layer
                    elif to_ == OUTPUT:
                        to_layer = POOLING_NAME

                    descriptor.connect_layers(last_out, to_layer)

        descriptor.connect_layers(POOLING_NAME, OUTPUT_NAME)

        return descriptor

    def plot(self, ax=None, alpha=1.0, layers_as_names=False, show=False, node_size=150, width=1.0, font_size=12, w_off=0.25, h_off=1, elevate_input=False):
        import matplotlib.pyplot as plt

        def my_layout(G, paths):
            nodes = G.nodes
            lengths = [-len(x) for x in paths]
            sorted_ = np.argsort(lengths)

            positions = dict()
            h = 0
            w = 0

            h_d = h_off
            w_d = w_off
            for index in sorted_:
                h = 0
                added = False
                path = paths[index]
                for node in path:
                    if node not in positions:
                        positions[node] = (w, h)
                        added = True
                        h -= h_d
                    else:
                        if h > positions[node][1]:
                            h = positions[node][1]

                if added:
                    if w >= 0:
                        w += w_d
                    w *= -1

            h = 0
            for node in nodes:
                if node not in positions:
                    positions[node] = (w, h)
                    h -= 1

            # if OUTPUT in positions:
            #     # positions[OUTPUT] = (positions[OUTPUT][0],
            #     #                      positions[OUTPUT][1]-1)
            #     positions[OUTPUT] = (positions[OUTPUT][0],
            #                          -10)
            if INPUT in positions and elevate_input:
                positions[INPUT] = (positions[INPUT][0], positions[INPUT][1]+1)

            left = dict()
            right = dict()
            for node in positions:
                w, h = positions[node]
                if w < 0:
                    left[h] = node
                elif w > 0:
                    right[h] = node

            h = 0
            for height in sorted(list(left.keys()), reverse=True):
                node = left[height]
                positions[node] = (positions[node][0], h)
                h -= 1
            h = 0
            for height in sorted(list(right.keys()), reverse=True):
                node = right[height]
                positions[node] = (positions[node][0], h)
                h -= 1
            return positions

        G = self.to_networkx()
        if ax is None:
            plt.figure()
        in_path = self.get_direct_paths()
        pos = my_layout(G, in_path)
        labels = {x: str(x).split('/')[-1] for x in G.nodes}
        pooling_key = list(self.hyperparameters.genes.keys())[-1]
        labels[INPUT] = 'INPUT'
        labels[OUTPUT] = 'OUTPUT'
        if layers_as_names:
            old_labels = labels
            old_labels.pop(INPUT)
            old_labels.pop(OUTPUT)
            labels = {x: layers_list[self.nodes.genes[int(
                x)].value[0]] for x in old_labels.keys()}
            labels[INPUT] = 'INPUT'
            labels[OUTPUT] = pooling_list[self.hyperparameters.genes[pooling_key].value[0]]

        nx.draw(G, labels=labels, pos=pos, node_color='w', with_labels=True,
                alpha=alpha, node_shape='s',
                node_size=node_size, width=width, font_size=font_size)

        nodes = set()
        for p in in_path:
            for node in p:
                nodes.add(node)
        nx.draw_networkx_nodes(G,  # labels=labels,
                               pos=pos,
                               node_color='w',
                               nodelist=list(nodes),
                               # with_labels=True,
                               alpha=alpha,
                               node_shape='s',
                               node_size=node_size,
                               # font_size=font_size
                               )
        if show:
            plt.show()

    def to_networkx(self, active_only=True, layers_as_names=False, remove_inactives=True):

        G = nx.DiGraph()
        genemap = dict()
        i = 0
        for g in sorted(self.connections.genes):
            gene = self.connections.genes[g]
            if gene.enabled or not active_only:
                for v in gene.value:
                    if v not in genemap.keys():
                        genemap[v] = str(i)
                        i += 1

        for g in self.connections.genes:
            gene = self.connections.genes[g]
            if gene.enabled or not active_only:
                if layers_as_names:
                    from_ = gene.value[0]
                    to_ = gene.value[1]
                    if from_ not in [INPUT, OUTPUT]:
                        from_ = genemap[from_]+'/'+layers_list[int(
                            self.nodes.genes[from_].value[0])]
                    if to_ not in [INPUT, OUTPUT]:
                        to_ = genemap[to_]+'/' + \
                            layers_list[int(self.nodes.genes[to_].value[-1])]
                    G.add_edge(from_, to_)
                else:
                    G.add_edge(*gene.value)

        if remove_inactives:
            nodes = set()
            paths = None
            paths = nx.all_simple_paths(G, INPUT, OUTPUT)

            for p in paths:
                for node in p:
                    nodes.add(node)
            all_nodes = list(G.nodes)
            for node in all_nodes:
                if node not in nodes:
                    G.remove_node(node)

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
