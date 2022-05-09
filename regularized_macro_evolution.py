"""
Created on 2018-12-17 19:09:40

@author: George Kyriakides
          ge.kyriakides@gmail.com
"""

import sys
import networkx


from nord.utils import count_parameters, get_genomes_from_logger, get_logger
from nord.neural_nets import LocalEvaluator, NeuralNet
from nord.configurations.all import Configs

from nord.design.metaheuristics.evolutionary.arc.arc_config import \
    CHANNELS_NO
from nord.design.metaheuristics.evolutionary.arc import (
    Genome, Innovation)
import torch.optim
import torch
import numpy as np
from copy import deepcopy
import traceback
import argparse
from mpi4py import MPI


parser = argparse.ArgumentParser(
    description='Regularized evolution for Macro-NAS.')

parser.add_argument('EPOCHS', type=int,
                    help='number of evaluation epochs')

parser.add_argument('--hot', action="store_true", dest='HOT_START', default=False,
                    help='hot start from the logs')

args = parser.parse_args()

EPOCHS = args.EPOCHS
comm = MPI.COMM_WORLD


worker = comm.rank
workers_no = comm.size

np.random.seed(1337+worker)
generations = 1000

population_size = 100
parents_no = 25
identity_prob = 0.00
add_node_prob = 1.0
initial_add_node_prob = 1.0

dataset = 'fashion-mnist'
fitness_key = 'accuracy'


config = Configs()
INPUT_SHAPE = config.INPUT_SHAPE[dataset]

innv = Innovation()
innv.new_generation()

main_logger_name = 'LOGS_ARC'+str(EPOCHS)+'epochs_'+str(comm.rank)
my_log = get_logger(main_logger_name)
error_log = get_logger('Errors_ARC'+str(EPOCHS)+'epochs_'+str(comm.rank))
info_log = get_logger('Infos_ARC'+str(EPOCHS)+'epochs_'+str(comm.rank))

history = []

evaluator = LocalEvaluator(torch.optim.Adam, {}, False)
# torch.optim.Adam, {'lr': 2.4e-2, 'weight_decay': 5e-4}, False)
population = []


def evaluate(g):
    global history
    d = g.to_descriptor(dimensions=len(INPUT_SHAPE))
    info_log.info(g.__repr__())
    fitness = 0
    total_time = 0
    try:
        # fitness = len(d.layers) + len(d.connections)
        loss, fitness, total_time = evaluator.descriptor_evaluate(
            d, EPOCHS, data_percentage=1, dataset=dataset)
        # THIS LINE CAUSES MANY PROBLEMS IN DISTRIBUTED SETTINGS!!!!! torch.cuda.empty_cache()

    except Exception as e:
        print('INVALID')
        print(d)
        tb = traceback.format_exc()
        error_log.info(tb)
        traceback.print_exc()

    history.append(fitness)
    return fitness, total_time


for i in range(population_size):
    g = Genome(identity_prob, add_node_prob,
               channels=CHANNELS_NO, strides=1, innovation=innv)
    # for j in range(4):
    #     g.mutate(add_node_rate=initial_add_node_prob)
    g.remove_recursions()
    population.append(g)

population = comm.bcast(population)
generation = 0

my_chunk = int(population_size/workers_no)
start = worker*my_chunk
end = population_size if worker == workers_no-1 else start+my_chunk
fitnesses = np.zeros(population_size)

log_population_data = []
if args.HOT_START:
    log_population_data = get_genomes_from_logger(main_logger_name)
    print(len(log_population_data))

if args.HOT_START:
    while(len(log_population_data) > 0):
        genome_data = log_population_data.pop(0)
        i = int(genome_data[0])
        generation = int(genome_data[1])
        fitness = float(genome_data[2])
        new_g = Genome.__from_repr__(genome_data[3])
        new_g.connections.index.pop(0)
        max_innv = max(max(new_g.connections.index), max(new_g.nodes.index))
        innv.inno_number = max(innv.inno_number, max_innv)
        new_g.innovation = innv
        np.random.seed(int(genome_data[5]))
        population[i] = new_g
        fitnesses[i] = fitness
        start += 1

print(fitnesses)
if generation == 0:
    for i in range(start, end):
        g = population[i]
        d = g.to_descriptor(dimensions=len(INPUT_SHAPE))
        fitness, total_time = evaluate(g)
        tag = ('_individual_%d' % i)+'_gen_%d' % generation
        # my_log.info((tag, fitness, g.__repr__(), total_time))
        my_log.info((i, generation, fitness, g.__repr__(),
                     total_time, np.random.get_state()[1][0]))
        fitnesses[i] = fitness[fitness_key]


fitnesses = comm.allreduce(fitnesses)

for i in range(population_size):
    population[i].fitness = fitnesses[i]
# if worker == 0:
#     print(fitnesses)


while(len(log_population_data) > 0):
    innv.new_generation()
    genome_data = log_population_data.pop(0)
    i = int(genome_data[0])
    generation = int(genome_data[1])
    fitness = float(genome_data[2])
    offspring = Genome.__from_repr__(genome_data[3])
    np.random.seed(int(genome_data[5]))
    offspring.fitness = fitness
    for w in range(workers_no):
        offspr = comm.bcast(offspring, root=w)
        population.pop(0)
        population.append(offspr)
        max_innv = max(max(offspr.connections.index), max(offspr.nodes.index))
        innv.inno_number = max(innv.inno_number, max_innv)
print('GENERATION:', generation)

while generation < generations/workers_no:

    innv.new_generation()
    generation += 1

    sample = np.random.choice(population, size=parents_no)
    parent = np.argmax([x.fitness for x in sample])
    parent = sample[parent]
    offspring = deepcopy(parent)
    offspring.mutate()
    i = worker
    info_log.info(offspring.__repr__())
    fitness, total_time = 0, 0
    try:
        offspring.remove_recursions()
        fitness, total_time = evaluate(offspring)

    except Exception as e:
        print('INVALID')
        d = g.to_descriptor(dimensions=len(INPUT_SHAPE))
        print(d)
        tb = traceback.format_exc()
        error_log.info(tb)
        traceback.print_exc()

    offspring.fitness = fitness[fitness_key]

    tag = ('_individual_%d' % i)+'_gen_%d' % generation
    # my_log.info((tag, fitness, offspring.__repr__()))
    my_log.info((i, generation, fitness, offspring.__repr__(),
                 total_time, np.random.get_state()[1][0]))
    # print(tag, fitness, offspring.__repr__())

    for w in range(workers_no):
        offspr = comm.bcast(offspring, root=w)
        population.pop(0)
        population.append(offspr)

# plt.figure()
# plt.plot(history)
# plt.show()
