import os.path
import traceback
from copy import deepcopy

import dgl
import networkx as nx
import numpy as np
import torch
import torch.utils.data
from dgl.data.utils import load_graphs, load_info, save_graphs, save_info
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset
from nord.utils import (assure_reproducibility, get_genomes_from_logger,
                        index_to_lower_triangle_index, graph_collate, graph_pair_collate)

DATA_ROOT = './data'


# class GraphTuple(tuple):
#     def to(self, device):
#         new_tuple = []
#         for g in self:
#             new_tuple.append(g.to(device))
#         return GraphTuple(new_tuple)


class GraphDataset(Dataset):

    @classmethod
    def __from_file__(cls, path, all_vs_all):
        ds = GraphDataset([], [], 3, all_vs_all)
        ds.load(path)
        ds.graphs_number = len(ds.labels)
        return ds

    def __init__(self, graphs, labels, classes, all_vs_all=True, e=0.003):
        """
        Args

        """
        self.graphs = graphs
        self.labels = labels
        self.classes = classes

        self.graphs_number = len(self.labels)
        self.e = e  # e% difference

        self.all_vs_all = all_vs_all

    def __len__(self):
        if self.classes == 1:
            return self.graphs_number

        if not self.all_vs_all:
            return self.graphs_number*(self.graphs_number+1)//2

        return self.graphs_number**2

    def __getitem__(self, idx):

        id_1 = idx % self.graphs_number
        id_2 = idx // self.graphs_number

        if not self.all_vs_all:
            # Sample only the lower triangle+diagonal
            id_1, id_2 = index_to_lower_triangle_index(idx)
            # k = np.floor(-0.5 + (np.sqrt(0.25+2*idx)))
            # j = idx - k*(k+1)/2
            # id_1 = int(k)
            # id_2 = int(j)

        label = None

        if self.classes == 3:

            label = [0, 1, 0]
            if self.labels[id_1] < self.labels[id_2]-self.e:
                label = [1, 0, 0]
            elif self.labels[id_1] > self.labels[id_2]+self.e:
                label = [0, 0, 1]
            # label = -1 if self.labels[id_1] < self.labels[id_2] else 1

        elif self.classes == 2:
            label = [1, 0] if self.labels[id_1] < self.labels[id_2] else [0, 1]
        elif self.classes == 1:
            return self.graphs[idx], self.labels[idx]

        return (self.graphs[id_1], self.graphs[id_2], label)

    def save(self, save_path):
        # save graphs and labels
        graph_path = os.path.join(save_path + '_dgl_graph.bin')
        save_graphs(graph_path, list(self.graphs), {
                    'labels': torch.Tensor(self.labels)})
        # save other information in python dict
        info_path = os.path.join(save_path + '_info.pkl')
        save_info(info_path, {'num_classes': self.classes})

    def load(self, save_path):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(save_path + '_dgl_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        info_path = os.path.join(save_path + '_info.pkl')
        self.classes = load_info(info_path)['num_classes']

    def has_cache(self, save_path):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(save_path + '_dgl_graph.bin')
        info_path = os.path.join(save_path + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)


def __get_fashion_mnist_architectures_sets(test_perc, val_perc, classes, all_vs_all=[False, True], group_same=False):

    LAYER_TYPES = 3

    from nord.design.metaheuristics.evolutionary.arc import Genome
    from nord.design.metaheuristics.evolutionary.arc.arc_config import (INPUT,
                                                                        OUTPUT)
    from sklearn.preprocessing import OneHotEncoder

    log_population_data = get_genomes_from_logger(
        'LOGS_ARC108epochs_3layertypes_all', logs_path='data/f-mnist_archs')
    population = []
    fitnesses = []

    while(len(log_population_data) > 0):
        genome_data = log_population_data.pop(0)
        fitness = float(genome_data[2])*100
        new_g = Genome.__from_repr__(genome_data[-3])
        # 0 index is None
        new_g.connections.index.pop(0)
        new_g.fitness = fitness
        population.append(new_g)
        fitnesses.append(fitness)

    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(np.array(
        [x for x in range(-2, LAYER_TYPES)]).reshape(-1, 1))

    trainset, valset, testset = None, None, None
    assure_reproducibility()
    graphs = []
    fits = []
    hashes = {}
    graph_no = 0
    graph_i = 0
    for g, f in zip(population, fitnesses):
        if f < 0.01:
            continue
        g_nx = g.to_networkx()
        labels = list(g_nx.nodes)

        g_nx = nx.relabel.convert_node_labels_to_integers(g_nx)
        new_labels = list(g_nx.nodes)

        g_dgl = dgl.from_networkx(g_nx)

        for i in range(len(new_labels)):
            data = g.nodes.genes[labels[i]].value
            if labels[i] == INPUT:
                data = -2
            elif labels[i] == OUTPUT:
                data = -1

            onehot_encoded = onehot_encoder.transform(
                np.array(data).reshape(-1, 1))
            g_dgl.nodes[new_labels[i]].data['x'] = torch.tensor(
                onehot_encoded)

        if group_same:
            g_nx = g_dgl.to_networkx(node_attrs=['x'])
            wl_hash = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(
                g_nx, node_attr='x', iterations=10)

            if wl_hash in hashes:
                hashes[wl_hash].append(graph_i)
            else:
                hashes[wl_hash] = [graph_i]

        graph_i += 1

        graphs.append(g_dgl)
        fits.append(f)

    if group_same:
        print('All graph instances: ', len(graphs))
        new_graphs = []
        new_fits = []

        for h, v in hashes.items():
            new_graphs.append(graphs[v[0]])
            new_fits.append(np.mean([fits[i] for i in v]))

        graphs = new_graphs
        fits = new_fits

    graph_no = len(graphs)
    print('Dataset graph instances: ', len(graphs))

    array_holder = np.empty(len(graphs), dtype=np.object)
    for i in range(len(graphs)):
        array_holder[i] = graphs[i]
    graphs = array_holder

    graphs = np.array(graphs)
    fits = np.array(fits)
    indices = np.random.permutation(graph_no)

    train_no = int(graph_no*(1-val_perc-test_perc))
    val_no = int(graph_no*val_perc)

    print(test_perc)
    print(train_no)
    trainset = GraphDataset(graphs[indices[:train_no]],
                            fits[indices[:train_no]],
                            classes, all_vs_all=all_vs_all[0])
    valset = None
    if (val_no > 0):
        valset = GraphDataset(graphs[indices[train_no:train_no+val_no]],
                              fits[indices[train_no:train_no+val_no]],
                              classes, all_vs_all=all_vs_all[1], e=1.00)  # e=0.3 as accs are in range 0-100

    testset = GraphDataset(graphs[indices[train_no+val_no:]],
                           fits[indices[train_no+val_no:]],
                           classes, all_vs_all=all_vs_all[1], e=1.00)  # e=0.3 as accs are in range 0-100

    print('{0} TRAIN {1} TEST'.format(
        len(trainset.graphs), len(testset.graphs)))

    return trainset, valset, testset


def get_fmnist_graphs(batch_size, test_perc, val_perc, classes, all_vs_all=[False, True]):
    from nord.utils import graph_collate, graph_pair_collate
    from torch.utils.data import DataLoader

    np.random.seed(1337)
    torch.random.manual_seed(1337)

    trainset, valset, testset = __get_fashion_mnist_architectures_sets(
        test_perc, val_perc, classes, all_vs_all=all_vs_all)

    train_loader, test_loader = None, None
    if classes > 1:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                  collate_fn=graph_pair_collate)

        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                                 collate_fn=graph_pair_collate)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                  collate_fn=graph_collate)

        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                                 collate_fn=graph_collate)

    return train_loader, test_loader, classes


def get_nasbench_graph_data_loaders(batch_size, initial_pop, mutated, classes, val_perc,
                                    test_perc, all_vs_all=[False, False]):

    np.random.seed(1337)
    torch.random.manual_seed(1337)

    trainset, valset, testset = __get_nasbench_graph_dataset(initial_pop, mutated, classes,  1337, val_perc,
                                                             test_perc, all_vs_all)

    train_loader, test_loader = None, None
    if classes > 1:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                  collate_fn=graph_pair_collate)

        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                                 collate_fn=graph_pair_collate)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                  collate_fn=graph_collate)

        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                                 collate_fn=graph_collate)

    return train_loader, test_loader, classes


def __get_nasbench_graph_dataset(initial_population_size=10, evolutions=10, classes=3,
                                 seed=1337, val_perc=0.1, test_perc=0.3, all_vs_all=[True, True],
                                 as_dgl=True, one_hot=True, DEBUG=False):

    from nord.design.metaheuristics.evolutionary.graph_rem.innovation import \
        Innovation
    from nord.design.metaheuristics.evolutionary.graph_rem.nasbench_genome import \
        Genome
    from nord.design.metaheuristics.evolutionary.graph_rem.rem_config import (
        INPUT, OUTPUT)
    from nord.neural_nets.evaluators.nasbench101_evaluator import \
        NASBench_101Evaluator

    evaluator = NASBench_101Evaluator()

    def __descriptor_to_feature_tensor(descriptor, one_hot=True):
        max_nodes = 7

        labels = ['input',
                  'conv1x1-bn-relu',
                  'conv3x3-bn-relu',
                  'maxpool3x3',
                  'output']
        # encoding = {labels[i]: eye[i,:] for i in range(5)}
        encoding = {labels[i]: i for i in range(5)}

        matrix, ops = NASBench_101Evaluator.descriptor_to_matrix(descriptor)
        ops_encoded = [encoding[key] for key in ops]
        nodes = len(ops)

        needed_nodes = max_nodes-nodes

        matrix = np.insert(
            matrix, [-1 for _ in range(needed_nodes)], 0, axis=1)
        matrix = np.insert(
            matrix, [-1 for _ in range(needed_nodes)], 0, axis=-0)

        feature_matrix = np.zeros((1, max_nodes, max_nodes))
        ops_encoded = ops_encoded[:-1]
        ops_encoded.extend(['' for _ in range(needed_nodes)])
        ops_encoded.append(4)

        if one_hot:
            feature_matrix = np.zeros((5, max_nodes, max_nodes))
            inds = np.where(matrix == 1)
            for i, j in zip(*inds):
                feature_matrix[ops_encoded[j], i, j] = 1

        else:
            inds = np.where(matrix == 1)
            for i, j in zip(*inds):
                feature_matrix[0, i, j] = ops_encoded[j]

        return feature_matrix

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # parents_perc = 0.25
    identity_prob = 0.05
    add_node_prob = 0.25
    initial_add_node_prob = 1.0

    innv = Innovation()
    innv.new_generation()

    population = []

    def evaluate(g):

        fitness = 0
        total_time = 0
        fits = []
        times = []
        try:
            d = g.to_descriptor()
            # fitness = len(d.layers) + len(d.connections)

            for _ in range(30):
                fitness, total_time = evaluator.descriptor_evaluate(
                    d, acc='test_accuracy')
                fits.append(fitness)
                times.append(total_time)
        except Exception as e:
            if DEBUG:
                print('INVALID')
                print(offspring)
                print(e)
                traceback.print_exc()

        if len(fits) == 0:
            return 0, 0
        fitness = np.average(fits)
        total_time = np.average(times)
        return fitness, total_time

    def in_dataset(g):
        try:
            d = g.to_descriptor()
        except Exception as e:
            if DEBUG:
                print('INVALID')
                print(offspring)
                print(e)
                traceback.print_exc()
            return False

        return evaluator.has_been_evaluated(d)

    for i in range(initial_population_size):
        g = Genome(identity_prob, add_node_prob, innovation=innv)
        g.mutate(add_node_rate=initial_add_node_prob)
        population.append(g)

    generation = 0
    fitnesses = []

    for i in range(initial_population_size):
        g = population[i]
        fitness, total_time = evaluate(g)
        fitnesses.append(fitness)

    while generation < evolutions:

        innv.new_generation()
        generation += 1

        # sample_sz = int(parents_perc*len(population))
        sample_sz = 1  # random search
        sample = np.random.choice(population, size=sample_sz)
        parent = np.argmax([x.fitness for x in sample])
        parent = sample[parent]
        offspring = deepcopy(parent)
        offspring.mutate()
        fitness, total_time = 0, 0

        if in_dataset(offspring):
            generation -= 1
            continue
            print('In Dataset')
        try:
            fitness, total_time = evaluate(offspring)

        except Exception as e:
            if DEBUG:
                print('INVALID')
                print(offspring)
                print(e)
                traceback.print_exc()
        if fitness < 0.01:
            generation -= 1
            continue

        # print(generation, fitness)
        offspring.fitness = fitness
        population.append(offspring)
        fitnesses.append(fitness)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(np.array(
        [x for x in range(-2, 3)]).reshape(-1, 1))

    trainset, valset, testset = None, None, None

    graphs = []
    fits = []
    graph_no = 0
    for g, f in zip(population, fitnesses):
        if f < 0.01:
            continue
        if as_dgl:
            gg = g.to_networkx()
            labels = list(gg.nodes)

            gg = nx.relabel.convert_node_labels_to_integers(gg)
            new_labels = list(gg.nodes)
            g_dgl = dgl.from_networkx(gg)

            for i in range(len(new_labels)):
                data = g.nodes.genes[labels[i]].value
                if labels[i] == INPUT:
                    data = -2
                elif labels[i] == OUTPUT:
                    data = -1

                onehot_encoded = onehot_encoder.transform(
                    np.array(data).reshape(-1, 1))
                g_dgl.nodes[new_labels[i]].data['x'] = torch.tensor(
                    onehot_encoded)
            graphs.append(g_dgl)
            fits.append(f)

        else:
            d = g.to_descriptor()
            ft = __descriptor_to_feature_tensor(d, one_hot)
            graphs.append(ft)
            fits.append(f)

    graph_no = len(graphs)

    if as_dgl:

        array_holder = np.empty(len(graphs), dtype=np.object)
        for i in range(len(graphs)):
            array_holder[i] = graphs[i]
        graphs = array_holder

    graphs = np.array(graphs)
    fits = np.array(fits)
    indices = np.random.permutation(graph_no)

    train_no = int(graph_no*(1-val_perc-test_perc))
    val_no = int(graph_no*val_perc)

    trainset = GraphDataset(graphs[indices[:train_no]],
                            fits[indices[:train_no]],
                            classes, all_vs_all=all_vs_all[0])
    valset = None
    if (val_no > 0):
        valset = GraphDataset(graphs[indices[train_no:train_no+val_no]],
                              fits[indices[train_no:train_no+val_no]],
                              classes, all_vs_all=all_vs_all[1])

    testset = GraphDataset(graphs[indices[train_no+val_no:]],
                           fits[indices[train_no+val_no:]],
                           classes, all_vs_all=all_vs_all[1])

    return trainset, valset, testset
