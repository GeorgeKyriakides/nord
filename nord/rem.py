"""
Created on 2018-12-17 19:09:40

@author: George Kyriakides
          ge.kyriakides@gmail.com
"""


from mpi4py import MPI
import traceback
import sys
from copy import deepcopy

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim

from architectural_design.metaheuristics.evolutionary.rem import (Genome,
                                                                  Innovation)
from configs import INPUT_SHAPE
from neural_nets import LocalEvaluator
from utils import get_logger

from architectural_design.metaheuristics.evolutionary.arc.rem_config import CHANNELS_NO


EPOCHS = int(sys.argv[1])

comm = MPI.COMM_WORLD


worker = comm.rank
workers_no = comm.size

np.random.seed(1337+worker)
generations = 1000

population_size = 100
parents_no = 25
identity_prob = 0.05
add_node_prob = 0.25
initial_add_node_prob = 1.0

dataset = 'fashion-mnist'

innv = Innovation()
innv.new_generation()


my_log = get_logger('LOGS_ARC'+str(EPOCHS)+'epochs_'+str(comm.rank))
error_log = get_logger('Errors_ARC'+str(EPOCHS)+'epochs_'+str(comm.rank))
info_log = get_logger('Infos_ARC'+str(EPOCHS)+'epochs_'+str(comm.rank))
history = []

evaluator = LocalEvaluator(torch.optim.Adam, {}, False)
# torch.optim.Adam, {'lr': 2.4e-2, 'weight_decay': 5e-4}, False)
population = []


def evaluate(g):
    global history
    d = g.to_descriptor(dimensions=len(INPUT_SHAPE[dataset]))

    info_log.info(g.__repr__())
    fitness = 0
    try:
        # fitness = len(d.layers) + len(d.connections)
        fitness = evaluator.descriptor_evaluate(
            d, EPOCHS, data_percentage=1, dataset=dataset)[1]
        torch.cuda.empty_cache()

    except Exception as e:
        print('INVALID')
        print(d)
        tb = traceback.format_exc()
        error_log.info(tb)
        traceback.print_exc()

    history.append(fitness)
    return fitness


for i in range(population_size):
    g = Genome(identity_prob, add_node_prob,
               channels=CHANNELS_NO, strides=1, innovation=innv)
    for j in range(4):
        g.mutate(add_node_rate=initial_add_node_prob)
    g.remove_recursions()
    population.append(g)

population = comm.bcast(population)
generation = 0

my_chunk = int(population_size/workers_no)
start = worker*my_chunk
end = population_size if worker == workers_no-1 else start+my_chunk
fitnesses = np.zeros(population_size)

for i in range(start, end):
    g = population[i]
    d = g.to_descriptor(dimensions=len(INPUT_SHAPE[dataset]))
    fitness = evaluate(g)
    # g.fitness = fitness
    fitnesses[i] = fitness
    tag = ('_individual_%d' % i)+'_gen_%d' % generation
    my_log.info((tag, fitness, g.__repr__()))


fitnesses = comm.allreduce(fitnesses)


for i in range(population_size):
    population[i].fitness = fitnesses[i]
# if worker == 0:
#     print(fitnesses)


while generation < generations/workers_no:

    innv.new_generation()
    generation += 1

    sample = np.random.choice(population, size=parents_no)
    parent = np.argmax([x.fitness for x in sample])
    parent = sample[parent]
    offspring = deepcopy(parent)
    offspring.mutate()
    offspring.remove_recursions()

    i = worker

    fitness = evaluate(offspring)
    offspring.fitness = fitness

    tag = ('_individual_%d' % i)+'_gen_%d' % generation
    my_log.info((tag, fitness, offspring.__repr__()))
    # print(tag, fitness, offspring.__repr__())

    for w in range(workers_no):
        offspr = comm.bcast(offspring, root=w)
        population.remove(0)
        population.append(offspr)

# plt.figure()
# plt.plot(history)
# plt.show()
