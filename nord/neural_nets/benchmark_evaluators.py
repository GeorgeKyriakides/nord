import os
import traceback

import networkx as nx
from nasbench import api
from nord.utils import pdownload

NASBENCH_TFRECORD = './data/nasbench_only108.tfrecord'
file_url = 'https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'


class BenchmarkEvaluator():
    """A class to evaluate a network on a benchmark
       NAS dataset.
    """

    def get_available_ops(self) -> list:
        return [CONV1X1, CONV3X3, MAXPOOL3X3]

    def __init__(self):
        if not os.path.isfile(NASBENCH_TFRECORD):
            print('Downloading NASBench-101 Data.')
            pdownload(file_url, NASBENCH_TFRECORD)
            print('Downloaded')

        self.dataset = api.NASBench(NASBENCH_TFRECORD)
        self.checked_models = {}

    def __descriptor_to_spec(self, descriptor):

        matrix, ops = BenchmarkEvaluator.descriptor_to_matrix(descriptor)
        try:
            model_spec = api.ModelSpec(
                # Adjacency matrix of the module
                matrix=matrix,
                # Operations at the vertices of the module, matches order of matrix
                ops=ops)
        except Exception:
            print(matrix)
            print(ops)
            print(descriptor)
            input('PROLBEM')
            traceback.print_exc()
            return None

        return model_spec

    def has_been_evaluated(self, descriptor):
        model_spec = self.__descriptor_to_spec(descriptor)
        return model_spec in self.checked_models

    def descriptor_evaluate(self, descriptor, acc='validation_accuracy'):

        model_spec = self.__descriptor_to_spec(descriptor)
        data = 0, 0
        try:

            hash_ = model_spec.hash_spec(self.dataset.config['available_ops'])
            if hash_ in self.checked_models:
                data = self.checked_models[hash_]
            else:
                data = self.dataset.query(model_spec)
                self.checked_models[hash_] = data
        except:
            traceback.print_exc()
            return 0, 0
        return data[acc], data['training_time']

    @staticmethod
    def descriptor_to_matrix(descriptor):

        graph = nx.DiGraph()

        for origin in descriptor.connections:
            for destination in descriptor.connections[origin]:
                graph.add_edge(origin, destination)

        node_lvls = {}
        first_node = min(list(descriptor.layers.keys()))
        last_node = max(list(descriptor.layers.keys()))
        nodes_no = len(list(descriptor.layers.keys()))

        for node in list(descriptor.layers.keys()):
            if node in (first_node, last_node):
                continue
            paths = nx.all_simple_paths(graph, source=node, target=last_node)
            lengths = [len(p) for p in paths]
            lengths.append(0)
            node_lvl = nodes_no - max(lengths)
            if node_lvl not in node_lvls:
                node_lvls[node_lvl] = [node]
            else:
                node_lvls[node_lvl].append(node)

        nodes_ordered = []
        ops = []

        first_lvl = -1
        last_lvl = nodes_no + 1
        try:
            node_lvls[first_lvl] = [first_node]
            node_lvls[last_lvl] = [last_node]

            for node_lvl in sorted(node_lvls):
                nodelist = node_lvls[node_lvl]
                nodes_ordered.extend(nodelist)
                for node in nodelist:
                    ops.append(descriptor.layers[node][0])

            matrix = nx.linalg.adjacency_matrix(
                graph, nodelist=nodes_ordered).todense().tolist()
        except Exception:
            print(nodes_ordered)
            print(descriptor)
            traceback.print_exc()
            return None

        return matrix, ops
