"""
Created on 2018-10-29

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

from copy import deepcopy
import networkx as nx
import torch.nn as nn
import numpy as np


from nord.neural_nets import NeuralDescriptor
from nord.neural_nets.layers import Identity, Conv2d151

from nord.utils import get_random_value

from .blueprint_chromosome import BlueprintChromosome
from .blueprint_gene import BlueprintConnectionGene, BlueprintLayerGene
from .arc_config import OUTPUT, OUTPUT_NAME, INPUT, INPUT_NAME, UNEVALUATED_FITNESS


class BlueprintGenome(object):

    def __init__(self, identity_rate, add_node_rate, modules_number,
                 innovation=None):

        self.connections = BlueprintChromosome()
        self.nodes = BlueprintChromosome()
        self.layer_bouds = [[0], [modules_number]]
        self.layer_bounds_types = [int]
        self.identity_rate = identity_rate
        self.add_node_rate = add_node_rate
        self.modules_number = modules_number

        self.fitness = UNEVALUATED_FITNESS

        # Add initial structure
        start_node = BlueprintLayerGene(None, io_node=True)
        end_node = BlueprintLayerGene(None, io_node=True)
        start_node.innovation_number = INPUT
        end_node.innovation_number = OUTPUT
        self.nodes.add_gene(start_node)
        self.nodes.add_gene(end_node)

        connection_node = BlueprintConnectionGene(
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

            new_node = BlueprintLayerGene(self.modules_number)
            self.innovation.assign_number(new_node)

            new_start = BlueprintConnectionGene(
                start_node, new_node.innovation_number)
            self.innovation.assign_number(new_start)

            new_end = BlueprintConnectionGene(
                new_node.innovation_number, end_node)
            self.innovation.assign_number(new_end)

            self.connections.add_gene(new_start)
            self.connections.add_gene(new_end)
            self.nodes.add_gene(new_node)

        elif r > identity_rate + add_node_rate:
            # Mutate a node or a connection
            r_t = get_random_value()

            if r_t < 0.5:  # Mutate a node
                g = np.random.choice(list(self.nodes.genes.keys()))
                gene = self.nodes.genes[g]
                gene.mutate(self.modules_number, 1.0)

            else:  # Mutate a connection
                g = np.random.choice(list(self.connections.genes.keys()))
                gene = self.connections.genes[g]
                gene.mutate(1.0)
                self.connections.genes[g] = gene
                # start_node, end_node = gene.value

                # nodes = list(self.nodes.genes)
                # nodes.remove(INPUT)
                # nodes.remove(OUTPUT)

                # end_node = np.random.choice(nodes+[OUTPUT])
                # gene.value = (start_node, end_node)

    def __repr__(self):
        return str({'Connections': self.connections, 'Nodes': self.nodes})

    def __hash__(self):
        return hash(self.connections) + hash(self.nodes)

    def __eq__(self, other):
        """Overrides the default implementation"""

        if isinstance(other, BlueprintGenome):

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
        g = BlueprintGenome([int],
                            [[0],
                             [1]], [], [],
                            0.1, 0.1, 0.1, None)
        rpr = ast.literal_eval(rpr)
        connections = rpr['Connections']
        for innovation in connections:
            g.connections.genes[innovation] = BlueprintConnectionGene.__from_repr__(
                connections[innovation])
            g.connections.index.append(innovation)

        g.connections.genes.pop(None)

        nodes = rpr['Nodes']
        for innovation in nodes:
            g.nodes.genes[innovation] = BlueprintLayerGene.__from_repr__(
                nodes[innovation])
            g.nodes.index.append(innovation)

        return g

    def to_descriptor(self, modules_list, dimensions=2):

        assert dimensions == 2

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
                    selected_module = gene.value
                    module = modules_list[selected_module]
                    module_descriptor = module.to_descriptor()
                    module_descriptor.add_suffix('_'+innv)
                    descriptor.layers.update(module_descriptor.layers)
                    descriptor.incoming_connections.update(
                        module_descriptor.incoming_connections)
                    descriptor.connections.update(
                        module_descriptor.connections)


        # Add IO layers
        descriptor.add_layer(Identity, {}, name=INPUT_NAME)
        descriptor.add_layer(Identity, {}, name=OUTPUT_NAME)
        descriptor.first_layer = INPUT_NAME
        descriptor.last_layer = OUTPUT_NAME

        # Connect the layers
        for g in self.connections.genes:
            gene = self.connections.genes[g]
            from_, to_ = gene.value
            # Connect all active
            if gene.enabled:
                # Only connecitons from/to active nodes should be added
                if from_ in actives and to_ in actives:
                    from_name = OUTPUT_NAME+'_'+str(from_)
                    to_name = INPUT_NAME+'_'+str(to_)

                    if from_ == INPUT:
                        from_name = INPUT_NAME
                    elif from_ == OUTPUT:
                        from_name = OUTPUT_NAME

                    if to_ == INPUT:
                        to_name = INPUT_NAME
                    elif to_ == OUTPUT:
                        to_name = OUTPUT_NAME
                    descriptor.connect_layers(from_name, to_name)

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

            if OUTPUT in positions:
                positions[OUTPUT] = (positions[OUTPUT][0],
                                     positions[OUTPUT][1]-1)
            if INPUT in positions:
                positions[INPUT] = (positions[INPUT][0], positions[INPUT][1]+1)
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
