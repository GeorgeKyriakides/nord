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

    def mutate(self, probability):
        if len(self.index) > 0:
            ln = len(self.index)
            g = np.random.randint(ln)
            g = self.index[g]
            gene = self.genes[g]
            gene.mutate(probability)

    def __repr__(self):
        return str(self.genes)
