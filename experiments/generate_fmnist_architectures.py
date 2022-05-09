"""
Created on 2018-12-17 19:09:40

@author: George Kyriakides
          ge.kyriakides@gmail.com
"""


import time
import traceback
from copy import deepcopy

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim

from nord.design.metaheuristics.genetics.neat import (
    Genome, Innovation)
from nord.configurations.all import Configs
from nord.neural_nets import LocalEvaluator, NeuralNet
from nord.utils import count_parameters, get_logger

np.random.seed(654321)


add_node_rate = 0.005
add_connection_rate = 0.01
mutation_rate = 0.5
crossover_rate = 0.75
generations = 51
nodes_limit = 500
active_nodes_limit = 20
parameters_limit = 10*1e6

dataset = 'fashion-mnist'
configs = Configs()

# no_filters, dropout_rate, weight_scaling, kernel_size, max_pooling
layer_bound_types = [int, float, float, int, bool]
layer_bounds = [[8, 0.0, 1.0, 1, 0],
                [8, 0.7, 1.0, 3, 1]]

# learning_rate, momentum, hue_shift, saturation/value_shift, sat/value_scale, cropped_img_size, spatial_scaling, random_horiz_flips, variance_norm, nesterov
hyperparameters_bound_types = [float, float, int,
                               float, float, int, float, bool, bool, bool]
hyperparameters_bounds = [[0.0001, 0.68, 0, 0.0, 0.0, 26, 0.0, 0, 0, 0],
                          [0.1, 0.99, 45, 0.5, 0.5, 32, 0.3, 1, 1, 1]]

representations = set()

innv = Innovation()
innv.new_generation()
genomes = []


g = Genome(layer_bound_types,
           layer_bounds,
           add_node_rate, add_connection_rate,
           mutation_rate, innv)
my_log = get_logger('architectures')

optimizers = {
    #   'Adadelta': [torch.optim.Adadelta, {}],
    'Adam': [torch.optim.Adam, {}],
    #   'Adamax': [torch.optim.Adamax, {}],
    # 'RMSprop': [torch.optim.RMSprop, {}],
    #   'SGD': [torch.optim.SGD, {'lr': 0.01, 'momentum': 0.0, 'nesterov': False}],
    #   'SGD_mom': [torch.optim.SGD, {'lr': 0.01, 'momentum': 0.9, 'nesterov': False}],
    #   'SGD_mom_nest': [torch.optim.SGD, {'lr': 0.01, 'momentum': 0.9, 'nesterov': True}]
}

evaluators = {key: LocalEvaluator(
    optimizers[key][0], optimizers[key][1], False) for key in optimizers}

n = 0

pop = []
generation = 0
repr = g.__repr__()
actives = set()
old_actives = []
while generation < generations-1:

    generation += 1
    old_g = deepcopy(g)

    while list(actives) in old_actives or repr in representations:

        innv.new_generation()
        g.mutate()
        g.remove_recursions()
        repr = g.__repr__()
        d = g.to_descriptor(
            dimensions=len(configs.INPUT_SHAPE[dataset]))
        actives = g.actives

    print('Generated', actives)

    repr = g.__repr__()
    g_eval = deepcopy(g)
    g_eval.remove_recursions()
    d = g_eval.to_descriptor(
        dimensions=len(configs.INPUT_SHAPE[dataset]))

    model = NeuralNet(
        d, configs.NUM_CLASSES[dataset], configs.INPUT_SHAPE[dataset], configs.CHANNELS[dataset], untrained=False)
    print(model)

    if count_parameters(model) > parameters_limit:
        print('Exceeded max parameters')
        g = np.random.choice(pop)
        d = g.to_descriptor(dimensions=len(configs.INPUT_SHAPE[dataset]))
        generation -= 1
        representations.add(repr)
        continue

    if g_eval.active_nodes > active_nodes_limit:
        print('Exceeded max active nodes')
        g = np.random.choice(pop)
        d = g.to_descriptor(dimensions=len(configs.INPUT_SHAPE[dataset]))
        generation -= 1
        representations.add(repr)
        continue

    print('Nodes: ', g_eval.active_nodes, ' Gen: ',
          n, 'Parameters: ', count_parameters(model))

    try:
        for opt_key in optimizers:
            optimizer = optimizers[opt_key]
            params = optimizer[1]

            optimizer = optimizer[0]
            ne = evaluators[opt_key]
            fitness = ne.descriptor_evaluate(
                d, 1, data_percentage=0.05, dataset=dataset)[1]
            # print('Allocated: ', torch.cuda.memory_allocated())
            # print('Cached: ', torch.cuda.memory_cached())
            print('Fitness:', fitness)
            # del ne
            # THIS LINE CAUSES MANY PROBLEMS IN DISTRIBUTED SETTINGS!!!!! torch.cuda.empty_cache()

    except Exception as e:
        print('INVALID')
        print(repr)
        traceback.print_exc()
        g = np.random.choice(pop)
        d = g.to_descriptor(dimensions=len(configs.INPUT_SHAPE[dataset]))
        # print(list(g.actives))
        generation -= 1
        representations.add(repr)
        continue

    old_actives.append(list(actives))
    pop.append(deepcopy(g))
    # g.connections.fitness = fitness
    # g.nodes.fitness = fitness

    representations.add(repr)
    n += 1
    my_log.info('"'+repr+'"')

    # g.remove_recursions()
    # g.plot()
    if len(g.nodes.genes) > nodes_limit:
        break
