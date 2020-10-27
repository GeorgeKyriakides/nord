"""
Created on 2018-09-02

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

# Simple example of finding the parameters for a simple ConvNet.


from design.problem_definitions import ConvNetSizeProblem
from utils import get_logger
import pygmo as pg

gens = 10
pop_size = 10

logger = get_logger('Local')


def main(untrained):
    pop = None
    prob = ConvNetSizeProblem(2, untrained=untrained)
    pg.set_global_rng_seed(94874)
    prob = pg.problem(prob)
    algo = pg.algorithm(pg.sga(1, m=0.2, mutation='uniform'))
    pop = pg.population(prob, pop_size)
    algo.set_verbosity(1)
    for g in range(gens):
        f = pop.get_f()
        x = pop.get_x()
        for i in range(len(x)):
            logger.info(str(g)+';'+str(x[i])+';'+str(f[i][0]))
        pop = algo.evolve(pop)
