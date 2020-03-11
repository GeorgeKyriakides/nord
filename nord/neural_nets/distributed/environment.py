"""
Created on 2018-08-06

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import horovod.torch as hvd
from mpi4py import MPI
from utils import get_logger

env = None


def close_workers():
    hvd.shutdown()
    MPI.Finalize()


class Environment():
    """The main context manager for the distributed environment.
    """

    def __init__(self, logger=None):
        self.local_rank = 0
        self.world_size = 0
        self.world_rank = 0
        self.comm = None
        logger = 'NORD' if logger is None else logger
        self.logger = get_logger(logger)

    def __enter__(self):
        global env
        hvd.init()
        self.local_rank = hvd.local_rank()
        self.world_size = hvd.size()
        self.world_rank = hvd.rank()
        self.comm = MPI.COMM_WORLD
        self.print_statuses()
        env = self
        return env

    def __exit__(self, type, value, traceback):
        global env
        self.barrier()
        close_workers()
        env = None

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
            for i in range(1, self.world_size):
                status = MPI.Status()
                msg = self.comm.recv(source=i, tag=MPI.ANY_TAG, status=status)
                print(msg)

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
