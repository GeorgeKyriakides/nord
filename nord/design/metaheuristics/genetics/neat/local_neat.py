"""
Created on 2018-09-02

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

# Simple example of finding the parameters for a simple ConvNet.

import numpy as np
from design.problem_definitions import ConvNetSizeProblem
from utils import get_logger
try:
    from mpi4py import MPI
except Exception:
    warnings.warn('mpi4py not found')


class DistributedGenetic():
    """Simple genetic algorithm with distributed individual evaluation
    """

    def __init__(self, population_size, generations,
                 crossover=0.9, mutation_prob=0.2,
                 untrained=True):
        self.pop_size = population_size
        self.gens = generations
        self.cr = crossover
        self.mt = mutation_prob

        self.population_x = []
        self.population_f = np.array([0.0 for i in range(population_size)])
        self.prob = ConvNetSizeProblem(
            2, distributed=False, untrained=untrained)
        self.comm = MPI.COMM_WORLD

        self.logger = get_logger(str(self.comm.Get_rank()))
        if self.comm.Get_rank() == 0:
            self.exp_logger = get_logger('Distributed_Genetic')
        self.init_population()

    def init_population(self):
        lb, ub = self.prob.get_bounds()
        for _ in range(self.pop_size):
            individual = []
            for i in range(len(ub)):
                individual.append(np.random.randint(lb[i], ub[i]))
            self.population_x.append(individual)

    def run(self):
        for gen in range(self.gens):
            if self.comm.Get_rank() == 0:

                self.evaluate_population_master()
            else:

                self.evaluate_population_worker()

            self.comm.Barrier()
            self.evolve_population()
            if self.comm.Get_rank() == 0:
                print('Best f for gen '+str(gen)+': ' +
                      str(np.max(self.population_f)))
                self.logger.info(str(self.population_x))

    def evolve_population(self):
        # Probabilities for roulette wheel selection
        probs = self.population_f/np.sum(self.population_f)
        new_pop = []
        for i in range(self.pop_size):
            # Crossover
            if i < self.cr*self.pop_size:
                p1 = np.random.choice(self.pop_size, size=1, p=probs)[0]
                p2 = np.random.choice(self.pop_size, size=1, p=probs)[0]
                p1, p2 = self.population_x[p1], self.population_x[p2]
                cpoint = np.random.randint(1, len(p1))
                offspring = p1[:cpoint]
                offspring.extend(p2[cpoint:])

                new_pop.append(self.mutate(offspring))
            # Don't crossover
            else:
                p1 = np.random.choice(self.pop_size, size=1, p=probs)[0]
                p1 = self.population_x[p1]
                new_pop.append(self.mutate(p1))
        self.population_x = new_pop
        self.population_f = np.array([0.0 for i in range(self.pop_size)])

    def mutate(self, individual):
        if np.random.uniform() < self.mt:
            mpoint = np.random.randint(0, len(individual))
            lb, ub = self.prob.get_bounds()
            new_val = np.random.randint(lb[mpoint], ub[mpoint])
            individual[mpoint] = new_val
        return individual

    def evaluate_population_master(self):
        world_sz = self.comm.Get_size()
        worker = 1
        # Send the population's parameters to the workers
        for i in range(self.pop_size):
            x = self.population_x[i]
            self.comm.send([i, x], dest=worker)
            worker = (worker+1) % world_sz
            if worker == 0:
                worker = 1

        # Wait for results
        for i in range(self.pop_size):
            status = MPI.Status()
            self.logger.info('Master waiting '+str(i))
            f = self.comm.recv(status=status)
            self.logger.info('Master got '+str(i))
            individual = status.Get_tag()
            self.exp_logger.info(
                str(self.population_x[individual])+';'+str(-f[0]))
            self.population_f[individual] = -f[0]

    def evaluate_population_worker(self):
        world_sz = self.comm.Get_size() - 1
        my_rank = self.comm.Get_rank()
        work = []

        # Get the required work
        for _ in range(my_rank-1, self.pop_size, world_sz):
            [tag, x] = self.comm.recv(tag=MPI.ANY_TAG,
                                      source=0)
            self.logger.info('Worker got '+str(tag))

            work.append([x, tag])
        # Send results
        for x, tag in work:
            f = self.prob.fitness(x)
            self.logger.info('Worker sending '+str(tag))
            self.comm.send(f, 0, tag=tag)
            self.logger.info('Worker done sending '+str(tag))


def main(untrained):
    d = DistributedGenetic(10, 10, untrained=untrained)
    d.run()
