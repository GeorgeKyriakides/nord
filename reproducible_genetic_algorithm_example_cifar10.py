"""
    Example of a simple genetic algorithm based on DeepNEAT


    Miikkulainen, Risto, et al. "Evolving deep neural networks."
    Artificial Intelligence in the Age of Neural Networks and Brain Computing.
    Academic Press, 2019. 293-312.

"""

import time
import traceback

import numpy as np
import torch.optim

from nord.design.metaheuristics.genetics.neat import Genome, Innovation
from nord.neural_nets import LocalEvaluator
from nord.utils import assure_reproducibility

assure_reproducibility()

# Genetic Algorithm Parameters
add_node_rate = 1.0
add_connection_rate = 0.0
mutation_rate = 0.5
generations = 3
population_sz = 2
tournament_sz = 1

# Evaluation parameters
EPOCHS = 1
dataset = 'cifar10'  # can also be 'fashion-mnist'
output_file = '../results/genetic_cifar10.out'


def write_to_file(msg):
    with open(output_file, 'a') as f:
        f.write(msg)
        f.write('\n')


write_to_file('Generation_No, Individual_No, Fitness, Genome')

# no_filters, dropout_rate, weight_scaling, kernel_size, max_pooling
layer_bound_types = [int, float, float, int, bool]
layer_bounds = [[8, 0.0, 0, 1, 0],
                [16, 0.7, 2.0, 3, 1]]

evaluator = LocalEvaluator(torch.optim.Adam, {}, False)
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
    g.mutate()
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
                d = g.to_descriptor(dimensions=2)
                loss, fitness, total_time = evaluator.descriptor_evaluate(
                    d, EPOCHS, data_percentage=1, dataset=dataset)
                if type(fitness) is dict:
                    fitness = fitness['accuracy']
                cache[g] = fitness
            else:
                fitness = cache[g]

            g.connections.fitness = fitness
            g.nodes.fitness = fitness
            write_to_file(str((r, j, np.round(fitness, 3), g)))
            if fitness == 0:
                print(g.__repr__())
        except Exception:
            traceback.print_exc()
            print(g.__repr__())
            continue

    new_population = []
    # Offspring Generation
    for _ in range(population_sz):

        pool_1 = np.random.choice(
            population, size=tournament_sz, replace=False)
        pool_2 = np.random.choice(
            population, size=tournament_sz, replace=False)

        parent_1 = np.argmax([f.nodes.fitness for f in pool_1])
        parent_2 = np.argmax([f.nodes.fitness for f in pool_2])

        parent_1 = pool_1[parent_1]
        parent_2 = pool_2[parent_2]

        offspring_1 = parent_1.crossover(parent_2)
        offspring_1.mutate()

        new_population.append(offspring_1)

    population = new_population
