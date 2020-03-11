"""
Created on 2018-08-04

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import horovod.torch as hvd
from mpi4py import MPI
from utils import get_logger
from neural_nets.neural_evaluators import DistributedEvaluator
import neural_nets.distributed.defaults as defaults
import warnings


class Environment():
    """The main context manager for the distributed environment.
    """

    def __init__(self, optimizer=None, optimizer_params=None, logger=None):
        self.local_rank = 0
        self.world_size = 0
        self.world_rank = 0
        self.comm = None
        logger = 'NORD' if logger is None else logger
        self.logger = get_logger(logger)

        if optimizer is None or optimizer_params is None:
            warnings.warn(
                'Optimizer or its parameters are None, using defaults.')
            optimizer = defaults.OPTIMIZER
            optimizer_params = defaults.OPTIMIZER_PARAMS

        verbose = defaults.VERBOSE
        self.ev = DistributedEvaluator(optimizer, optimizer_params, verbose)

    def __enter__(self):

        hvd.init()
        self.local_rank = hvd.local_rank()
        self.world_size = hvd.size()
        self.world_rank = hvd.rank()
        self.comm = MPI.COMM_WORLD
        self.print_statuses()
        self.workers_wait()

        return self

    def __exit__(self, type, value, traceback):
        self.shutdown()
        del self

    def shutdown(self):
        """Master sends signal to workers to stop waiting for workload.
           All processes shut down horovod and finalize MPI.
        """
        if self.world_rank == 0:
            print('Master stopped')
            self.comm.bcast(True)  # Notify workers to stop
        print('Worker %d stopped' % self.world_rank)
        hvd.shutdown()
        MPI.Finalize()

    def workers_wait(self):
        """Workers wait for signal and start to evaluate the decsriptor if prompted,
           else they exit and start shutdown procedure.
        """
        if self.world_rank > 0:
            while(True):
                stop = False
                stop = self.comm.bcast(stop)
                if stop:
                    break

                descriptor = dict()
                descriptor = self.comm.bcast(descriptor)

                untrained = None
                untrained = self.comm.bcast(untrained)

                epochs = 0
                epochs = self.comm.bcast(epochs)

                data_percentage = 0
                data_percentage = self.comm.bcast(data_percentage)

                self.ev.descriptor_evaluate(descriptor, epochs, untrained)

    def descriptor_evaluate(self, descriptor, epochs=5,
                            data_percentage=1, untrained=False):
        # Broadcast False so workers will not break from workers_wait
        self.comm.bcast(False)

        descriptor = self.comm.bcast(descriptor)
        untrained = self.comm.bcast(untrained)
        epochs = self.comm.bcast(epochs)
        data_percentage = self.comm.bcast(data_percentage)
        return self.ev.descriptor_evaluate(descriptor, epochs,
                                           data_percentage, untrained)

    def print_statuses(self):
        """Print the rank of each process
        """
        if not self.master():
            self.comm.send(('Worker %d of %d (local %d)' %
                            (self.world_rank+1,
                             self.world_size,
                             self.local_rank)), 0)
        else:
            self.print('Master initialized.')
            self.log('Master initialized.')
            for i in range(1, self.world_size):
                status = MPI.Status()
                msg = self.comm.recv(source=i, tag=MPI.ANY_TAG, status=status)
                self.print(msg)
                self.log(msg)

    def barrier(self):
        """Implement a simple barrier
        (i.e. all workers must reach this line in order to continue)
        """
        self.comm.Barrier()

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
        if self.world_rank == 0:
            return True
        return False

    def log(self, msg):
        """Function to log a message to the log file.
        """
        if self.master():
            self.logger.info(msg)
