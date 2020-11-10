"""
Created on 2018-08-04

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

import time
import warnings
from concurrent import futures
from typing import Dict, List, Type

import numpy as np
import torch
import torch.nn as nn


from nord.configurations.all import Configs
from nord.neural_nets import NeuralDescriptor, NeuralNet
from nord.data_curators import (get_cifar10_distributed,
                                get_cifar10,
                                get_fashion_mnist,
                                get_activity_recognition_data)
from nord.utils import progress_bar

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

    def __init__(self, optimizer_class: Type,
                 optimizer_params: Dict, verbose: bool):

        (self.trainloader, self.trainsampler,
         self.testloader, self.testsampler,
         self.classes) = [None]*5
        self.data_loaded = False
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.verbose = verbose
        self.conf = Configs()

    def load_data(self, data_percentage: float):
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

    def get_optimizer(self, net: nn.Module):
        """Returns the optimizer to train the network
        """
        raise NotImplementedError

    def print_status_bar(self, batch_id: int, loss: float):
        """Print status after each epoch
        """
        raise NotImplementedError

    def descriptor_to_net(self, descriptor: NeuralDescriptor, untrained: bool):
        """Make a net from descriptor, accounting for any distributed comms
        """
        raise NotImplementedError

    def process_returns(self, loss: float, metrics: dict):
        """Process loss and metrics, accounting for any distributed testing
        """
        raise NotImplementedError

    def train(self, net: nn.Module, epochs: int):
        """Network training.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network.

        epochs : int
            The number of epochs to train the network.
        """

        device = self.set_device()

        try:
            print(MPI.COMM_WORLD.rank, 'Train Device: ', device)
        except Exception:
            print('Train Device: ', device)

        net.to(device)
        net.train()

        optimizer = self.get_optimizer(net)

        # for p in net.parameters():
        #     print(p)
        #     break

        criterion_loss = self.conf.CRITERION[self.dataset]()

        if self.conf.CRITERION[self.dataset] == nn.KLDivLoss:
            def criterion(x, y): return criterion_loss(x.float(), y.float())
        else:
            def criterion(x, y): return criterion_loss(x, y)

        for epoch in range(epochs):  # loop over the dataset multiple times
            if self.trainsampler is not None:
                print('TRAINSAMPLER NOT NONE')
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
        del optimizer

    def test(self, net: nn.Module):
        """Distributed network evaluation.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network.

        Returns
        -------
        test_metrics : float
            The average metrics.

        """
        device = self.set_device()

        try:
            print(MPI.COMM_WORLD.rank, 'Test Device: ', device)
        except Exception:
            print('Test Device: ', device)

        net.to(device)
        net.eval()

        test_loss = 0
        criterion_loss = self.conf.CRITERION[self.dataset]()

        if self.conf.CRITERION[self.dataset] == nn.KLDivLoss:
            def criterion(x, y): return criterion_loss(x.float(), y.float())
        else:
            def criterion(x, y): return criterion_loss(x, y)

        all_predicted = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                all_predicted.extend(outputs)
                all_targets.extend(targets)
                self.print_status_bar(batch_idx, test_loss)

        if len(self.testloader) > 0:
            test_loss /= len(self.testloader)
        if self.testsampler is not None:
            if len(self.testsampler) > 0:
                test_loss /= len(self.testsampler)

        metrics = {}

        all_predicted = torch.stack(all_predicted)
        all_targets = torch.stack(all_targets)

        for metric in self.conf.METRIC[self.dataset]:
            metrics.update(metric(all_predicted, all_targets))

        return test_loss, metrics

    def descriptor_evaluate(self, descriptor: NeuralDescriptor,  epochs: int,
                            data_percentage: float = 1.0,
                            untrained: bool = False, dataset: str = None):
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
        test_metrics : dict
            The average metrics.

        loss: float
            The value of the loss function.

        total_time:
            The time required to train

        """
        self.load_data(data_percentage, dataset)
        net = self.descriptor_to_net(descriptor, untrained)
        print(net)
        return self.net_evaluate(net,  epochs, data_percentage,
                                 untrained, dataset)

    def net_evaluate(self, net: nn.Module,  epochs: int,
                     data_percentage: float = 1.0, untrained: bool = False,
                     dataset: str = None, return_net=False):
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
        test_metrics : dict
            The average metrics across all workers.

        loss: float
            The value of the loss function.

        total_time:
            The time required to train

        """
        self.load_data(data_percentage, dataset)

        start_time = time.time()

        if not net.functional:
            metrics = {}
            for metric in self.conf.METRIC[self.dataset]:
                metrics.update(
                    metric(torch.Tensor([[1], [0]]), torch.Tensor([[-1], [2]]))
                )
            return 0, metrics, 0

        if not untrained:
            self.train(net, epochs)

        total_time = time.time() - start_time  # in seconds

        loss, metrics = self.test(net)

        loss, metrics = self.process_returns(loss, metrics)

        if not return_net:
            return loss, metrics, total_time
        else:
            return loss, metrics, total_time, net


class LocalEvaluator(AbstractNeuralEvaluator):

    def load_data(self, data_percentage: float, dataset: str):

        if not self.data_loaded:
            self.data_loaded = True
            self.dataset = dataset

            (self.trainloader,
                self.testloader,
                self.classes) = self.conf.DATA_LOAD[self.dataset](data_percentage)

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

    def get_optimizer(self, net: nn.Module):
        """Returns the optimizer to train the network
        """
        return self.optimizer_class(net.parameters(), **self.optimizer_params)

    def print_status_bar(self, batch_id: int, loss: float):
        """Print status after each epoch
        """
        if self.verbose:
            progress_bar(batch_id, len(self.trainloader), 'Loss: %.2f'
                         % (loss/(batch_id+1)))

    def descriptor_to_net(self, descriptor: NeuralDescriptor, untrained: bool):
        """Make a net from descriptor, accounting for any distributed comms
        """
        return NeuralNet(descriptor, self.conf.NUM_CLASSES[self.dataset],
                         self.conf.INPUT_SHAPE[self.dataset], self.conf.CHANNELS[self.dataset],
                         untrained=untrained,
                         keep_dimensions=self.conf.DIMENSION_KEEPING[self.dataset],
                         dense_part=self.conf.DENSE_PART[self.dataset])

    def process_returns(self, loss: float, metrics: dict):
        """Process loss and metrics, accounting for any distributed testing
        """
        if self.verbose:
            print('Metrics of the network on the test images: ', (
                metrics))
            print('Loss of the network on the test images: %.4f ' % (
                loss))
        return loss, metrics


class LocalBatchEvaluator(LocalEvaluator):

    def train_work(self, params: tuple):
        optimizer, criterion, net, inputs, labels, my_id = params
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        my_loss = loss.item()
        return my_id, my_loss

    def train(self, nets: List[nn.Module], epochs: int):

        device = self.set_device()

        print('Device: ', device)

        ex = futures.ThreadPoolExecutor(max_workers=4)

        optimizers = []
        for net in nets:
            net.train()
            net.to(device)

            optimizer = self.get_optimizer(net)
            optimizers.append(optimizer)

        t = time.time()
        criterion = self.conf.CRITERION[self.dataset]()
        for epoch in range(epochs):  # loop over the dataset multiple times
            if self.trainsampler is not None:
                self.trainsampler.set_epoch(epoch)

            test_losses = [0 for _ in range(len(nets))]
            for batch_idx, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                args = ((optimizers[i], criterion, nets[i],
                         inputs, labels, i) for i in range(len(nets)))

                results = ex.map(self.train_work, args)

                for i, res_loss in results:
                    # print(res_loss)
                    test_loss = test_losses[i]
                    test_losses[i] = test_loss + res_loss

                    # self.print_status_bar(batch_idx, test_loss)
            print(time.time()-t)
            # criterion.increase_threshold()

    def descriptors_evaluate(self, descriptors: List[NeuralDescriptor],
                             epochs: int, data_percentage: float = 1.0,
                             untrained: bool = False,
                             dataset: str = None):
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
        nets = []
        for descriptor in descriptors:
            net = self.descriptor_to_net(descriptor, untrained)
            nets.append(net)

        return self.nets_evaluate(nets,  epochs, data_percentage,
                                  untrained, dataset)

    def nets_evaluate(self, nets: List[nn.Module],  epochs: int,
                      data_percentage: float = 1.0, untrained: bool = False,
                      dataset: str = None):
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
        self.load_data(data_percentage, dataset)
        nets_no = len(nets)
        losses = np.zeros(nets_no)
        accuracies = np.zeros(nets_no)
        non_functionals = []

        start_time = time.time()
        for i in range(nets_no):
            net = nets[i]
            if not net.functional:
                non_functionals.append(i)
        if not untrained:
            self.train(nets, epochs)

        total_time = time.time() - start_time  # in seconds

        for i in range(nets_no):
            if i not in non_functionals:
                loss, accuracy = self.test(net)

                loss, accuracy = self.process_returns(loss, accuracy)
                losses[i] = loss
                accuracies[i] = accuracy

        return losses, accuracies, total_time


class DistributedEvaluator(AbstractNeuralEvaluator):

    def load_data(self, data_percentage: float, dataset: str):

        if not self.data_loaded:
            self.data_loaded = True
            if dataset == 'cifar10':
                (self.trainloader, self.trainsampler,
                 self.testloader, self.testsampler,
                 self.classes) = get_cifar10_distributed(hvd.size(),
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

    def get_optimizer(self, net: nn.Module):
        """Returns the optimizer to train the network
        """
        optimizer = self.optimizer_class(
            net.parameters(), **self.optimizer_params)
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=net.named_parameters())

        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        return optimizer

    def print_status_bar(self, batch_id: int, loss: float):
        """Print status after each epoch
        """
        if hvd.rank() == 0 and self.verbose:
            progress_bar(batch_id, len(self.trainloader), 'Loss: %.2f'
                         % (loss/(batch_id+1)))

    def descriptor_to_net(self, descriptor: NeuralDescriptor, untrained: bool):
        """Make a net from descriptor, accounting for any distributed comms
        """
        MPI.COMM_WORLD.barrier()
        descriptor = MPI.COMM_WORLD.bcast(descriptor)

        net = NeuralNet(descriptor, self.conf.NUM_CLASSES[self.dataset],
                        self.conf.INPUT_SHAPE[self.dataset], self.conf.CHANNELS[self.dataset],
                        untrained=untrained,
                        keep_dimensions=self.conf.DIMENSION_KEEPING[self.dataset])
        hvd.broadcast_parameters(net.state_dict(), root_rank=0)

        return net

    def process_returns(self, loss: float, accuracy: float):
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
