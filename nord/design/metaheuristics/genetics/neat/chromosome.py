"""
Created on 2018-10-29

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import numpy as np


class Chromosome(object):

    def __init__(self):
        self.genes = dict()
        self.fitness = None
        self.index = list()

    def set_fitness(self, fitness):
        self.fitness = fitness

    def add_gene(self, gene):
        self.genes[gene.innovation_number] = gene
        self.index.append(gene.innovation_number)

    def crossover(self, other):
        # Sort parents
        if self.fitness > other.fitness:
            p1, p2 = self, other
        else:
            p2, p1 = self, other

        offspring = Chromosome()
        for i in p1.genes:
            # Homologous genes
            if i in p2.index:
                offspring.genes[i] = p1.genes[i].crossover(p2.genes[i])
            # Else inherit from parent with probability to remain inactive
            # Call crossover with self for convenience
            else:
                new_gene = p1.genes[i].crossover(p1.genes[i])
                offspring.genes[i] = new_gene
        offspring.index = list(offspring.genes.keys())
        return offspring

    def mutate(self, probability):
        if len(self.index) > 0:
            ln = len(self.index)
            g = np.random.randint(ln)
            g = self.index[g]
            gene = self.genes[g]
            gene.mutate(probability)

    def __repr__(self):
        return str(self.genes)
