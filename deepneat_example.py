"""
Created on 2018-12-17 19:09:40

@author: George Kyriakides
          ge.kyriakides@gmail.com
"""


import time
import traceback

import torch.optim

from nord.configurations.all import Configs
from nord.design.metaheuristics.genetics.neat import Genome, Innovation
from nord.neural_nets import LocalEvaluator

add_node_rate = 0.03
add_connection_rate = 0.05
mutation_rate = 0.5
crossover_rate = 0.75
generations = 100
population_sz = 20
config = Configs()

EPOCHS = 1
dataset = 'cifar10'


# no_filters, dropout_rate, weight_scaling, kernel_size, max_pooling
layer_bound_types = [int, float, float, int, bool]
layer_bounds = [[32, 0.0, 0, 1, 0],
                [256, 0.7, 2.0, 3, 1]]

# learning_rate, momentum, hue_shift, saturation/value_shift, sat/value_scale, cropped_img_size, spatial_scaling, random_horiz_flips, variance_norm, nesterov
hyperparameters_bound_types = [float, float, int,
                               float, float, int, float, bool, bool, bool]

hyperparameters_bounds = [[0.0001, 0.68, 0, 0, 0, 26, 0, 0, 0, 0],
                          [0.1, 0.99, 45, 0.5, 0.5, 32, 0.3, 1, 1, 1]]

evaluator = LocalEvaluator(torch.optim.Adam, {}, True)
cache = dict()
i = Innovation()
i.new_generation()
population = []
# Population initialization
for _ in range(population_sz):
    g = Genome(layer_bound_types,
               layer_bounds,
               add_node_rate, add_connection_rate,
               mutation_rate, i)
    population.append(g)


for r in range(generations):

    t = time.time()
    i.new_generation()

    # Evaluation
    for j in range(len(population)):
        g = population[j]
        try:
            if g not in cache:
                print('Evaluating', g)
                d = g.to_descriptor(
                    dimensions=len(config.INPUT_SHAPE[dataset]))
                loss, fitness, total_time = evaluator.descriptor_evaluate(
                    d, EPOCHS, data_percentage=1, dataset=dataset)
                cache[g] = fitness
            else:
                fitness = cache[g]

            g.connections.fitness = fitness
            g.nodes.fitness = fitness
            print(('G', j, fitness))
            if fitness == 0:
                print(g.__repr__())
        except Exception as hh:
            print(('**Exception:', hh))

            traceback.print_exc()
            print(('******ERROR:', j))
            print(g.__repr__())
            continue
