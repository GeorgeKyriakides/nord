"""
Created on 2018-08-04

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

import warnings

import torch
import torch.nn as nn

from configs import CHANNELS, CRITERION, INPUT_SHAPE, NUM_CLASSES, PROBLEM_TYPE
from neural_nets.neural_builder import NeuralNet
from utils import progress_bar

try:
    import horovod.torch as hvd
    hvd_available = True
except Exception:
    hvd_available = False
    warnings.warn('Horovod not found')

try:
    from mpi4py import MPI
except Exception:
    warnings.warn('mpi4py not found')


class AbstractNeuralEvaluator():
    """A class to load a dataset and evaluate a network on it
       by distributing the load across N workers.
    """

    def __init__(self, optimizer_class, optimizer_params, verbose):

        (self.trainloader, self.trainsampler,
         self.testloader, self.testsampler,
         self.classes) = [None]*5
        self.data_loaded = False
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.verbose = verbose

    def load_data(self, data_percentage):
        """Check if the data is loaded and act accordingly.
           This is to make sure that the distributed environment
           has been initialized. Instantiate trainloader, trainsampler,
           testloader, testsampler, classes
        """
        raise NotImplementedError

    def set_device(self):
        """Returns the device that will run the experiment,
            usually cuda:dev_no or cpu

        """
        raise NotImplementedError

    def get_optimizer(self, net):
        """Returns the optimizer to train the network
        """
        raise NotImplementedError

    def print_status_bar(self, batch_id, loss):
        """Print status after each epoch
        """
        raise NotImplementedError

    def descriptor_to_net(self, descriptor, untrained):
        """Make a net from descriptor, accounting for any distributed comms
        """
        raise NotImplementedError

    def process_returns(self, loss, accuracy):
        """Process loss and accuracy, accounting for any distributed testing
        """
        raise NotImplementedError

    def train(self, net, epochs):
        """Network training.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network.

        epochs : int
            The number of epochs to train the network.
        """

        device = self.set_device()

        # print('Device: ', device)

        net.train()
        net.to(device)

        optimizer = self.get_optimizer(net)
        criterion = CRITERION[self.dataset]
        for epoch in range(epochs):  # loop over the dataset multiple times
            if self.trainsampler is not None:
                self.trainsampler.set_epoch(epoch)
            test_loss = 0
            for batch_idx, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                test_loss += loss.item()

                self.print_status_bar(batch_idx, test_loss)
            # criterion.increase_threshold()

    def test(self, net):
        """Distributed network evaluation.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network.

        Returns
        -------
        test_accuracy : float
            The average accuracy across all workers.

        """
        device = self.set_device()

        test_accuracy = 0
        net.eval()
        net.to(device)
        test_loss = 0
        correct = 0
        total = 0
        criterion = CRITERION[self.dataset]

        problem_type = PROBLEM_TYPE[self.dataset]

        with torch.no_grad():
            if problem_type == 'regression':
                for batch_idx, (inputs, targets) in enumerate(self.testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    self.print_status_bar(batch_idx, test_loss)
            else:
                for batch_idx, (inputs, targets) in enumerate(self.testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    total += targets.size(0)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                    self.print_status_bar(batch_idx, test_loss)

        if len(self.testloader) > 0:
            test_loss /= len(self.testloader)
        if self.testsampler is not None:
            if len(self.testsampler) > 0:
                test_loss /= len(self.testsampler)

        if PROBLEM_TYPE[self.dataset] == 'classification':
            test_accuracy = 100 * correct / total
        else:
            test_accuracy = -test_loss

        return test_loss, test_accuracy

    def descriptor_evaluate(self, descriptor,  epochs,
                            data_percentage=1, untrained=False,
                            dataset='cifar10'):
        """Distributed network evaluation, with a NeuralDescriptor input.

        Parameters
        ----------
        descriptor : NeuralDescriptor
            The neural network's descriptor object.

        untrained : bool (optional)
            If True, skip the training.

        epochs : int
            Number of epochs that the net will be trained.

        Returns
        -------
        test_accuracy : float
            The average accuracy across all workers.

        """
        self.dataset = dataset
        net = self.descriptor_to_net(descriptor, untrained)

        return self.net_evaluate(net,  epochs, data_percentage, untrained, dataset)

    def net_evaluate(self, net,  epochs,
                     data_percentage=1, untrained=False,
                     dataset='cifar10'):
        """Distributed network evaluation, with a NeuralDescriptor input.

        Parameters
        ----------
        descriptor : NeuralDescriptor
            The neural network's descriptor object.

        untrained : bool (optional)
            If True, skip the training.

        epochs : int
            Number of epochs that the net will be trained.

        Returns
        -------
        test_accuracy : float
            The average accuracy across all workers.

        """
        self.dataset = dataset
        self.load_data(data_percentage, dataset)

        if not net.functional:
            if PROBLEM_TYPE[self.dataset] == 'classification':
                return 0, 0
            else:
                return 1000, -1000
        if not untrained:
            self.train(net, epochs)

        loss, accuracy = self.test(net)

        loss, accuracy = self.process_returns(loss, accuracy)

        return loss, accuracy


class LocalEvaluator(AbstractNeuralEvaluator):

    def load_data(self, data_percentage, dataset):
        from .data_curators import get_cifar10_partial, get_climate_data_partial, get_fashion_mnist_partial
        if not self.data_loaded:
            self.data_loaded = True
            if dataset == 'cifar10':
                (self.trainloader,
                 self.testloader,
                 self.classes) = get_cifar10_partial(data_percentage)
            elif dataset == 'fashion-mnist':
                (self.trainloader,
                 self.testloader,
                 self.classes) = get_fashion_mnist_partial(data_percentage)
            else:
                (self.trainloader,
                 self.testloader,
                 self.classes) = get_climate_data_partial(data_percentage, prediction_window=NUM_CLASSES['time_series'])

    def set_device(self):
        """Returns the device that will run the experiment,
            usually cuda:dev_no or cpu

        """
        if not torch.cuda.is_available():
            return 'cpu'

        if hvd_available:
            try:
                return 'cuda:%d' % hvd.local_rank()  # If horovod is initialized
            except ValueError as e:
                print(e)
                warnings.warn('Horovod not initialized')
                try:
                    return 'cuda:%d' % MPI.COMM_WORLD.rank  # If mpi is running
                except Exception as ee:
                    print(ee)
                    warnings.warn(
                        'MPI not initialized, using one GPU per node.')

        return 'cuda:0'

    def get_optimizer(self, net):
        """Returns the optimizer to train the network
        """
        return self.optimizer_class(net.parameters(), **self.optimizer_params)

    def print_status_bar(self, batch_id, loss):
        """Print status after each epoch
        """
        if self.verbose:
            progress_bar(batch_id, len(self.trainloader), 'Loss: %.2f'
                         % (loss/(batch_id+1)))

    def descriptor_to_net(self, descriptor, untrained):
        """Make a net from descriptor, accounting for any distributed comms
        """
        return NeuralNet(descriptor, NUM_CLASSES[self.dataset],
                         INPUT_SHAPE[self.dataset], CHANNELS[self.dataset],
                         untrained=untrained, problem_type=PROBLEM_TYPE[self.dataset])

    def process_returns(self, loss, accuracy):
        """Process loss and accuracy, accounting for any distributed testing
        """
        if self.verbose:
            print('Accuracy of the network on the test images: %.2f %%' % (
                accuracy))
            print('Loss of the network on the test images: %.4f ' % (
                loss))
        return loss, accuracy


class DistributedEvaluator(AbstractNeuralEvaluator):

    def load_data(self, data_percentage, dataset):
        from .data_curators import get_cifar10_distributed_partial

        if not self.data_loaded:
            self.data_loaded = True
            if dataset == 'cifar10':
                (self.trainloader, self.trainsampler,
                 self.testloader, self.testsampler,
                 self.classes) = get_cifar10_distributed_partial(hvd.size(),
                                                                 hvd.rank(),
                                                                 data_percentage)
            else:
                raise NotImplementedError

    def set_device(self):
        """Returns the device that will run the experiment,
            usually cuda:dev_no or cpu

        """
        if not torch.cuda.is_available():
            return 'cpu'

        return 'cuda:%d' % hvd.local_rank()

    def get_optimizer(self, net):
        """Returns the optimizer to train the network
        """
        optimizer = self.optimizer_class(
            net.parameters(), **self.optimizer_params)
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=net.named_parameters())

        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        return optimizer

    def print_status_bar(self, batch_id, loss):
        """Print status after each epoch
        """
        if hvd.rank() == 0 and self.verbose:
            progress_bar(batch_id, len(self.trainloader), 'Loss: %.2f'
                         % (loss/(batch_id+1)))

    def descriptor_to_net(self, descriptor, untrained):
        """Make a net from descriptor, accounting for any distributed comms
        """
        MPI.COMM_WORLD.barrier()
        descriptor = MPI.COMM_WORLD.bcast(descriptor)

        net = NeuralNet(descriptor, NUM_CLASSES[self.dataset],
                        INPUT_SHAPE[self.dataset], CHANNELS[self.dataset],
                        untrained=untrained, problem_type=PROBLEM_TYPE[self.dataset])
        hvd.broadcast_parameters(net.state_dict(), root_rank=0)

        return net

    def process_returns(self, loss, accuracy):
        """Process loss and accuracy, accounting for any distributed testing
        """
        loss = MPI.COMM_WORLD.allreduce(loss)/hvd.size()
        accuracy = MPI.COMM_WORLD.allreduce(accuracy)/hvd.size()

        if hvd.rank() == 0 and self.verbose:

            print('Accuracy of the network on the test images: %.2f %%' % (
                accuracy))
            print('Loss of the network on the test images: %.4f ' % (
                loss))
        return loss, accuracy
