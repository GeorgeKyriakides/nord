"""
Created on 2018-08-04

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import os
import warnings
import torch

import torch.distributed as dist

from nord.neural_nets.distributed.distributed_configurator import DistributedConfig
from nord.neural_nets import DistributedEvaluator
from nord.utils import get_logger, singleton


OPTIMIZER = torch.optim.Adam
OPTIMIZER_PARAMS = {}
VERBOSE = True


class Environment():
    """The main context manager for the distributed environment.
    """

    def __init__(self, configuration: DistributedConfig, optimizer=None, optimizer_params=None, logger=None):

        self.configuration = configuration
        if self.configuration.world_rank == 0:
            print(self.configuration)

        logger = 'NORD' if logger is None else logger
        self.logger = get_logger(logger)

        if optimizer is None or optimizer_params is None:
            warnings.warn(
                'Optimizer or its parameters are None, using defaults.')
            optimizer = OPTIMIZER
            optimizer_params = OPTIMIZER_PARAMS

        verbose = VERBOSE
        self.ev = DistributedEvaluator(
            configuration, optimizer, optimizer_params, verbose)

    def __enter__(self):

        os.environ['MASTER_ADDR'] = self.configuration.master_address
        os.environ['MASTER_PORT'] = self.configuration.master_port
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = self.configuration.debug_level

        print('PG')
        # initialize the process group
        self.init_group()
        print('PS')
        self.print_statuses()
        print('WW')
        self.barrier()
        self.workers_wait()

        return self

    def init_group(self):
        dist.init_process_group(
            self.configuration.backend, rank=self.configuration.local_rank, world_size=self.configuration.world_size)

    def __exit__(self, type, value, traceback):
        dist.destroy_process_group()
        self.shutdown()
        del self

    def shutdown(self):
        """Master sends signal to workers to stop waiting for workload.
           All processes shut down .
        """
        if self.configuration.world_rank == 0:
            print('Master stopped')
            dist.broadcast_object_list([True], 0)  # Notify workers to stop
        print('Worker %d stopped' % self.configuration.world_rank)

    def workers_wait(self):
        """Workers wait for signal and start to evaluate the decsriptor if prompted,
           else they exit and start shutdown procedure.
        """
        if self.configuration.world_rank > 0:
            while(True):
                stop = False
                dist.broadcast_object_list([stop], 0)
                if stop:
                    break

                descriptor = dict()
                untrained = None
                epochs = 0
                data_percentage = 0

                dist.broadcast_object_list(
                    [descriptor, untrained, epochs, data_percentage], 0)

                self.ev.descriptor_evaluate(descriptor, epochs, untrained)

    def descriptor_evaluate(self, descriptor, epochs=5,
                            data_percentage=1, untrained=False, dataset='cifar10'):
        # Broadcast False so workers will not break from workers_wait
        dist.broadcast_object_list([False], 0)

        dist.broadcast_object_list(
            [descriptor, untrained, epochs, data_percentage], 0)
        return self.ev.descriptor_evaluate(descriptor, epochs,
                                           data_percentage, untrained, dataset)

    def print_statuses(self):
        """Print the rank of each process
        """
        if not self.master():
            dist.send(torch.tensor
                      ([self.configuration.world_rank+1,
                        self.configuration.world_size,
                        self.configuration.local_rank]), 0)
        else:
            self.print('Master initialized.')
            self.log('Master initialized.')
            for i in range(1, self.configuration.world_size):
                data = torch.tensor([-1, -2, -3])
                dist.recv(data, src=i)
                data = data.numpy()
                msg = f'Worker {data[0]} with local rank {data[2]} of {data[1]}'
                self.print(msg)
                self.log(msg)

    def barrier(self):
        """Implement a simple barrier
        (i.e. all workers must reach this line in order to continue)
        """
        dist.barrier()

    def print(self, msg):
        """Function to print a message only once
           (from master).
        """
        if self.master():
            print(msg)

    def master(self):
        """ Return True for the master process,
            False for all else.
        """
        if self.configuration.world_rank == 0:
            return True
        return False

    def log(self, msg):
        """Function to log a message to the log file.
        """
        if self.master():
            self.logger.info(msg)
