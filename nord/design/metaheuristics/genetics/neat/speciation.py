"""
Created on 2019-02-01 22:11:43

@author: George Kyriakides
          ge.kyriakides@gmail.com
"""
import copy
from typing import List

import numpy as np

from .genome import Genome


def get_distance(g1: Genome, g2: Genome):
    c_1 = 1
    c_2 = 1

    N = 1

    innovations_1, innovations_2 = [], []

    innovations_1.extend(g1.connections.index)
    innovations_1.extend(g1.nodes.index)

    innovations_2.extend(g2.connections.index)
    innovations_2.extend(g2.nodes.index)

    # Get the max innovation of the two genomes
    max_innovation_1 = max(innovations_1)
    max_innovation_2 = max(innovations_2)

    # Get the minimum of the above
    min_innovation = min(max_innovation_1, max_innovation_2)

    total_1 = len(innovations_1)
    total_2 = len(innovations_2)
    # Swap so innovations_1 has min_innovation
    if min_innovation == max_innovation_2:
        tmp = innovations_1
        innovations_1 = innovations_2
        innovations_2 = tmp

        tmp = total_1
        total_1 = total_2
        total_2 = tmp

    # Excess and disjoint
    E = 0
    D = 0

    for i in innovations_1:
        # Homologous
        if i in innovations_2:
            innovations_2.remove(i)
            total_2 -= 1

        else:
            D += 1

        total_1 -= 1

    for i in sorted(innovations_2):
        if i < min_innovation:
            D += 1
            total_2 -= 1
        else:
            break
    E = total_2

    delta = (c_1*E + c_2*D) / N
    return delta


def sharing_f(delta: float):
    threshold = 4
    if delta > threshold:
        return 0
    return 1


def get_distance_matrix(pop: List[Genome]):
    pop_size = len(pop)
    matrix = np.zeros((pop_size, pop_size))

    for i in range(pop_size):
        for j in range(i+1, pop_size):
            d = get_distance(pop[i], pop[j])
            matrix[i][j] = d
            matrix[j][i] = d

    return matrix


class SpeciesPopulations(object):

    def __init__(self, population_size: int, crossover_rate: float):
        self.species = list()
        self.population_size = population_size
        self.reproduction_sizes = []
        self.crossover_rate = crossover_rate

    def update_species(self, pop: List[Genome]):

        for s in self.species:
            s.population = []

        # count = 0
        for g in pop:
            # print(count)
            # count += 1
            found = False
            for i in range(len(self.species)):
                species = self.species[i]
                if species.attempt_append(g):
                    found = True
                    break

            if not found:
                s = Species(g)
                self.species.append(s)

        for s in self.species:
            s.share_fitness()

        self.reproduction_sizes = self.__get_reproduction_sizes()

        for i in range(len(self.species)):
            # print('species', i, len(self.species))
            self.species[i].reproduce(self.reproduction_sizes[i],
                                      self.crossover_rate)

    def get_all_individuals(self):
        individuals = []
        for s in self.species:
            individuals.extend(s.population)
        return individuals

    def __get_reproduction_sizes(self):

        fitness_sum = sum([s.total_fitness for s in self.species])
        reproduction_sizes = [
            int(np.floor(self.population_size*s.total_fitness/fitness_sum)) for s in self.species]

        return reproduction_sizes


class Species(object):

    def __init__(self, g: Genome):
        self.representative = g
        self.population = [g]
        self.total_fitness = 0

    def attempt_append(self, g: Genome):
        if sharing_f(get_distance(g, self.representative)) == 1:
            self.population.append(g)
            return True
        return False

    def share_fitness(self):
        self.total_fitness = 0
        sz = len(self.population)

        # count = 0
        for g in self.population:
            # print('share', count)
            # count += 1
            g.connections.fitness = g.connections.fitness/sz
            g.nodes.fitness = g.nodes.fitness/sz

            self.total_fitness += (g.connections.fitness + g.nodes.fitness)

        if len(self.population) > 0:
            self.representative = np.random.choice(self.population)

    def reproduce(self, new_size: int, crossover_rate: float):

        if len(self.population) > 0:
            # Crossover - Mutate
            new_genomes = []
            for j in range(new_size):
                # print(j, 'size:', new_size)
                a = self.tournament_selection()
                if np.random.uniform() < crossover_rate:
                    # print('Crossover')
                    b = self.tournament_selection()
                    g = a.crossover(b)
                else:
                    # print('Mutate')
                    g = copy.deepcopy(a)
                # print('Done')
                # print(g)
                g.mutate()
                # print('Append')
                new_genomes.append(g)

            self.population = new_genomes

    def tournament_selection(self):
        tournament_pc = 0.5
        pressure = 0.8

        tournament_sz = max(
            int(np.floor(len(self.population) * tournament_pc)), 1)
        t = np.random.choice(self.population, size=tournament_sz)
        fs = [p.connections.fitness for p in t]
        ranks = np.argsort(fs)

        place = np.random.uniform()
        cumm_p = 0

        for i in range(tournament_sz):
            cumm_p += pressure * ((1-pressure)**i)
            if place < cumm_p:
                return t[ranks[i]]

        return t[ranks[0]]
