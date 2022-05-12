"""
Created on 2018-09-02

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

# Simple example of finding the parameters for a small ConvNet.
# Local PyGMO implementation with distributed training.

import numpy as np
import pygmo as pg

from nord.design.pygmo_problem_definitions.conv_net_size_problem import \
    ConvNetSizeProblem
from nord.neural_nets.distributed.distributed_configurator import \
    DistributedConfig
from nord.neural_nets.distributed.environment import Environment

gens = 10
pop_size = 10


config = DistributedConfig('127.0.0.1', '1234', 1, 0)


def main(untrained):
    with Environment(config) as e:
        if e.master():
            pop = None
            prob = ConvNetSizeProblem(config=config, untrained=untrained)
            pg.set_global_rng_seed(45714)
            prob = pg.problem(prob)
            algo = pg.algorithm(pg.sga(1, m=0.2, mutation='uniform'))
            pop = pg.population(prob, pop_size)
            algo.set_verbosity(1)
            for g in range(gens):
                f = pop.get_f()
                x = pop.get_x()
                for i in range(len(x)):
                    e.log(
                        str(g)+';' +
                        np.array2string(x[i], precision=2,
                                        separator=',',
                                        suppress_small=True)+';' +
                        str(f[i][0]))
                pop = algo.evolve(pop)


if __name__ == '__main__':
    main(False)
