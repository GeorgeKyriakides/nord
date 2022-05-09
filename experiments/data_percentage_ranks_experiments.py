"""
Created on 2019-03-04 13:59:18

@author: George Kyriakides
          ge.kyriakides@gmail.com
"""


import time
import traceback

from nord.utils import get_logger
# import matplotlib.pyplot as plt
import numpy as np

from nord.design.metaheuristics.genetics.neat import (Genome,
                                                      Innovation,
                                                      SpeciesPopulations)
from nord.neural_nets import NeuralNet
from nord.neural_nets import LocalEvaluator

import torch.optim
from mpi4py import MPI


comm = MPI.COMM_WORLD


np.random.seed(123456)

dataset = 'cifar10'

add_node_rate = 0.05
add_connection_rate = 0.1
mutation_rate = 0.5
crossover_rate = 0.75
generations = 20

# no_filters, dropout_rate, weight_scaling, kernel_size, max_pooling
layer_bound_types = [int, float, float, int, bool]
layer_bounds = [[32, 0.0, 0, 1, 0],
                [256, 0.7, 2.0, 3, 1]]


data_percentages = [(i+1)/10 for i in range(10)]
epochs_schedule = [1, 5, 10, 20, 50]

optimizer = torch.optim.RMSprop
params = {}
evaluator = LocalEvaluator(optimizer, params, False)
cache = dict()
innv = Innovation()
innv.new_generation()
genomes = []


g = Genome(layer_bound_types,
           layer_bounds,
           add_node_rate, add_connection_rate,
           mutation_rate, innv)

my_log = get_logger('LOGS_'+str(comm.rank))
perc_index = comm.rank
perc_sz = len(data_percentages)

for generation in range(generations):

    t = time.time()
    innv.new_generation()

    # Evaluation
    for i in range(len(epochs_schedule)):
        epochs = epochs_schedule[i]
        data_perc = data_percentages[perc_index]

        str_percentage = data_perc*100
        tag = ('%d_perc' % str_percentage)+('_epochs_%d' %
                                            epochs)+('_gen_%d' % generation)
        print(tag, g)
        d = g.to_descriptor((32, 32, 3))

        fitness = evaluator.descriptor_evaluate(
            descriptor=d, epochs=epochs, data_percentage=data_perc, dataset=dataset)[1]
        cache[g] = fitness

        g.connections.fitness = fitness
        g.nodes.fitness = fitness
        print(tag, fitness)

        my_log.info((tag, fitness, g.__repr__()))

    np.random.seed(123456+generation)
    g.mutate()

    # Cyclic roll for optimizers
    perc_index = (perc_index+1) % perc_sz


MPI.Finalize()
