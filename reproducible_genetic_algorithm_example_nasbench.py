"""
    Example of a simple genetic algorithm based on DeepNEAT


    Miikkulainen, Risto, et al. "Evolving deep neural networks."
    Artificial Intelligence in the Age of Neural Networks and Brain Computing.
    Academic Press, 2019. 293-312.

"""

import matplotlib as mpl
import time
from collections import namedtuple
from copy import deepcopy

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from nord.neural_nets.evaluators.nasbench101_evaluator import \
    NASBench_101Evaluator
from nord.neural_nets.neural_descriptor import NeuralDescriptor
from nord.utils import assure_reproducibility


evaluator = NASBench_101Evaluator()


ops = evaluator.get_available_ops()


add_node_prob = 0.5
add_connection_prob = 0.5


population = []

initial_population_size = 50
evolutions = 450
sample_sz = 25
max_layers = 5
available_layers = len(ops)

generation = 0
fitnesses = []


OUTPUT_LAYER_NAME = '9999'

assure_reproducibility()


def add_random_connection(descriptor: NeuralDescriptor):
    # Get a source, but don't get output as source node [:-1]
    from_ = np.random.choice(list(descriptor.connections.keys())[:-1])
    # NORD names layers with a number prefix. Get the source node number
    # and find possible output nodes that have a greater number as prefix
    from_number = from_.split('_')[0]
    avail_to_layers = [x for x in descriptor.layers.keys() if int(
        from_number) < int(x.split('_')[0])]
    # If there are available output layers, choose one at random and add the connection
    if len(avail_to_layers) > 1:
        to_ = np.random.choice(avail_to_layers)
        descriptor.connect_layers(from_, to_)

    return descriptor


def mutate(descriptor: NeuralDescriptor):
    r = np.random.uniform()
    try:
        # If we have the maximum number of hidden layers (+2 to account for input-output layers) we can only add connections
        if r < add_connection_prob or len(descriptor.layers) == max_layers+2:
            descriptor = add_random_connection(descriptor)
        # If we have less layers and the random number generator dictates to add
        # a layer, proceed
        elif r < add_node_prob+add_connection_prob:
            # Get source node, but again don't get the output layer as source node
            from_ = np.random.choice(list(descriptor.connections.keys())[:-1])
            # Get all of its connections and choose one at random to replace
            to_ = np.random.choice(list(descriptor.connections[from_]))
            # Get a random layer
            layer_index = np.random.choice(available_layers)
            # Name it with a prefix that is larger than the source node
            # and smaller than the destination node. Getting the average
            # of the source and destination prefixes is sufficient.
            layer_name = (int(from_.split('_')) + (to_.split('_')))/2
            layer_name = str(layer_name)+'_Layer'
            layer_name = descriptor.add_layer(ops[layer_index], {}, layer_name)

            # Disconnect the original source and destination nodes
            descriptor.disconnect_layers(from_, to_)
            # Connect them to the new layer
            descriptor.connect_layers(from_, layer_name)
            descriptor.connect_layers(layer_name, to_)
        # Ensure that the descriptor's last layer name points
        # to the actual output layer. Adding a layer
        # changes the last_layer name, so we have to reset it
        descriptor.last_layer = OUTPUT_LAYER_NAME
    except Exception:
        pass

    return descriptor


def evaluate(descriptor):

    fitness = 0
    total_time = 0
    try:
        fitness, total_time = evaluator.descriptor_evaluate(
            descriptor)
    except Exception as e:
        pass

    return fitness


Individual = namedtuple('Individual', 'descriptor fitness')

start_t = time.time()
for i in range(initial_population_size):
    d = NeuralDescriptor()
    this_layers = 1+np.random.choice(max_layers-1)  # at least 2 layers
    d.add_layer_sequential('input', {})
    for _ in range(this_layers+1):
        layer_index = np.random.choice(available_layers)
        layer_name = d.add_layer_sequential(ops[layer_index], {})
        if np.random.uniform() < 1/7:
            d = add_random_connection(d)

    d.add_layer_sequential('output', {}, OUTPUT_LAYER_NAME)
    f = evaluate(d)
    fitnesses.append(f)
    population.append(Individual(d, f))


while generation < evolutions:

    generation += 1

    sample = np.random.choice(len(population), size=sample_sz)

    sample = [population[x] for x in sample]
    fits = [x.fitness for x in sample]
    parent = np.argmax(fits)
    # parent = np.random.choice(
    #     [x for x in range(len(fits))], p=np.array(fits)/np.sum(fits))
    parent = sample[parent]
    offspring = deepcopy(parent)

    offspring = mutate(offspring.descriptor)

    fitness = evaluate(offspring)

    # print(generation, fitness)
    population.append(Individual(offspring, fitness))
    fitnesses.append(fitness)

    population.pop(0)


sns.set_style("white")
sns.set_style("ticks")
sns.set_palette('colorblind')
plt.rcParams["figure.dpi"] = 300


print('Total search time: ', time.time() - start_t)
print('Total evaluation time: ', evaluator.dataset.training_time_spent)
plt.plot([x if x > 0 else None for x in accs],
         label='Reinforcement Learning 34.1s', linestyle='--')
plt.plot([x if x > 0 else None for x in fitnesses], label='Evolutionary 0.8s')

plt.ylim((0.6, 0.96))
plt.xlabel('Network Evaluations')
plt.ylabel('Validation Accuracy')
plt.title('')
plt.legend(loc='lower right')
