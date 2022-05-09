"""
Created on 2020-09-07 15:28:28

@author: George Kyriakides
          ge.kyriakides@gmail.com
"""

import argparse
import os
import sys
import traceback
from copy import deepcopy

import numpy as np
import torch
import torch.optim
from mpi4py import MPI
from scipy.stats import kendalltau

from nord.design.metaheuristics.evolutionary.graph_rem import (
    Genome, Innovation)
from nord.data_curators.graphs import get_fmnist_graphs

from nord.design.metaheuristics.evolutionary.graph_rem.rem_config import \
    CHANNELS_NO
from nord.configurations.all import Configs
from nord.neural_nets.neural_builder import GraphRankNet
from nord.neural_nets.evaluators.neural_evaluators import LocalEvaluator
from nord.utils import count_parameters, get_genomes_from_logger, get_logger

# import matplotlib.pyplot as plt


try:
    import networkx
except:
    sys.path.insert(0,
                    '/users/pa19/gkyriak/networkx_installation/networkx')


configs = Configs()
dataset = 'graph-nasbench3'
fitness_key = 'kendall'

HOT_START = False
EPOCHS = 2
comm = MPI.COMM_WORLD

INPUT_SHAPE = configs.INPUT_SHAPE[dataset]

worker = comm.rank
workers_no = comm.size

np.random.seed(1337+worker)
generations = 400

population_size = 100
parents_no = 50
identity_prob = 0.5
add_node_prob = 0.25
initial_add_node_prob = 1.0


innv = Innovation()
innv.new_generation()


base_logger_name = dataset
main_logger_name = 'LOGS_ARC_'+base_logger_name + '_' +\
    str(EPOCHS)+'epochs_'+str(comm.rank)
my_log = get_logger(main_logger_name)
error_log = get_logger('Errors_ARC_'+base_logger_name+'_' +
                       str(EPOCHS)+'epochs_'+str(comm.rank))
info_log = get_logger('Infos_ARC_'+base_logger_name + '_' +
                      str(EPOCHS)+'epochs_'+str(comm.rank))


history = []


evaluator = LocalEvaluator(
    torch.optim.AdamW, {}, False)

##### EVALUATOR#####
population = []


def get_metric(genome_data, key):

    metrics = None
    for m in genome_data:
        if key in m:
            metrics = m
            break
    metrics = metrics.replace('[', '').replace(
        ']', '').replace('{', '').replace('}', '')
    split = metrics.split(':')
    if split[0].strip().replace("'", '') == key:
        return float(split[-1].split("=")[-1])
    return 0


def evaluate(g):
    global history
    d = g.to_descriptor(dimensions=len(INPUT_SHAPE))
    info_log.info(g.__repr__())
    fitness, total_time = {'acc': [0.3, 0.3], 'kendall': kendalltau(
        [1, 0], [0, 1]), 'top_kendall': kendalltau([1, 0], [0, 1])}, 0
    try:
        # g.plot()
        # plt.show()
        # fitness = len(d.layers) + len(d.connections)
        loss, fitness, total_time = evaluator.descriptor_evaluate(
            d, EPOCHS, data_percentage=1, dataset=dataset)
        # THIS LINE CAUSES MANY PROBLEMS IN DISTRIBUTED SETTINGS!!!!! torch.cuda.empty_cache()

    except Exception as e:
        print('INVALID')
        print(g)
        print(d)
        tb = traceback.format_exc()
        error_log.info(tb)
        traceback.print_exc()

    history.append(fitness)
    print(fitness)
    return fitness, total_time


for i in range(population_size):
    g = Genome(identity_prob, add_node_prob,
               channels=CHANNELS_NO, strides=1, innovation=innv)
    for j in range(1):
        g.mutate(add_node_rate=initial_add_node_prob)
    g.remove_recursions()
    population.append(g)

population = comm.bcast(population)
generation = 0

my_chunk = int(population_size/workers_no)
start = worker*my_chunk
end = population_size if worker == workers_no-1 else start+my_chunk
fitnesses = np.zeros(population_size)

log_population_data = []
if HOT_START:
    log_population_data = get_genomes_from_logger(main_logger_name)
    print(len(log_population_data))

if HOT_START:
    while(len(log_population_data) > 0):
        genome_data = log_population_data.pop(0)
        i = int(genome_data[0])
        generation = int(genome_data[1])
        fitness = get_metric(genome_data, fitness_key)
        new_g = Genome.__from_repr__(genome_data[-3])
        new_g.connections.index.pop(0)
        max_innv = max(max(new_g.connections.index), max(new_g.nodes.index))
        innv.inno_number = max(innv.inno_number, max_innv)
        new_g.innovation = innv
        new_g.fitness = fitness
        np.random.seed(int(genome_data[-1]))
        population[i] = new_g

        try:
            nn = GraphRankNet(new_g.to_descriptor(), 10, (5,))
            print(count_parameters(nn))
            # if new_g.fitness < 0.4:
            #     nn.plot()
            #     new_g.plot(show=True)

        except Exception as e:
            print('Logs Reloading', new_g, e)

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
        fitnesses[i] = fitness[fitness_key][0]


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

    fitness, total_time = {'acc': [0.3, 0.3], 'kendall': kendalltau(
        [1, 0], [0, 1]), 'top_kendall': kendalltau([1, 0], [0, 1])}, 0
    try:
        offspring.remove_recursions()
        fitness, total_time = evaluate(offspring)

    except Exception as e:
        print('INVALID')
        d = g.to_descriptor(dimensions=len(INPUT_SHAPE))
        print(g)
        print(d)
        tb = traceback.format_exc()
        error_log.info(tb)
        traceback.print_exc()

    offspring.fitness = fitness[fitness_key][0]

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
