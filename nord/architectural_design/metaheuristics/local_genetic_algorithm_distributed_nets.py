"""
Created on 2018-08-02

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

# Simple example of finding the parameters for a simple ConvNet.

import pygmo as pg
from .conv_net_size_problem import ConvNetSizeProblem
from neural_nets.distributed import Environment

gens = 10
pop_size = 10


def main(untrained):
    with Environment() as e:
        pop = None
        prob = ConvNetSizeProblem(2, distributed=True, untrained=untrained)
        pg.set_global_rng_seed(45714)
        prob = pg.problem(prob)
        algo = pg.algorithm(pg.sga(1, m=0.2, mutation='uniform'))
        pop = pg.population(prob, pop_size)
        algo.set_verbosity(1)
        for g in range(gens):
            f = pop.get_f()
            x = pop.get_x()
            for i in range(len(x)):
                e.log(str(g)+';'+str(x[i])+';'+str(f[i][0]))
            pop = algo.evolve(pop)
