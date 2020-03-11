"""
Created on 2018-08-06

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

import torch
import torch.nn as nn
import torch.optim as optim
import horovod.torch as hvd
from mpi4py import MPI
from utils import progress_bar
from .data_curator import get_cifar10_distributed
from neural_nets.neural_builder import NeuralNet


criterion = nn.CrossEntropyLoss()
EPOCHS = 5
momentum = 0.5
initial_lr = 0.01
final_lr = initial_lr


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")


class DistributedNeuralEvaluator(object):
    """A class to load a dataset and evaluate a network on it
       by distributing the load across N workers.
    """

    def __init__(self):

        (self.trainloader, self.trainsampler,
         self.testloader, self.testsampler,
         self.classes) = [None]*5
        self.data_loaded = False

    def __load_data__(self):
        """Check if the data is loaded and act accordingly.
           This is to make sure that the distributed environment
           has been initialized.
        """
        global final_lr
        final_lr = initial_lr * hvd.size()
        if not self.data_loaded:
            self.data_loaded = True
            (self.trainloader, self.trainsampler,
             self.testloader, self.testsampler,
             self.classes) = get_cifar10_distributed(hvd.size(),
                                                     hvd.rank())

    def train(self, net, epochs, verbose):
        """Distributed network training.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network.

        epochs : int
            The number of epochs to train the network.

        verbose : bool
            If True, a progress bar will be displayed.

        """
        net.train()
        net.to(device)

        optimizer = optim.SGD(net.parameters(), lr=final_lr,
                              momentum=momentum)
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=net.named_parameters())
        for epoch in range(epochs):  # loop over the dataset multiple times
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
                # print statistics
                if verbose:
                    progress_bar(batch_idx, len(self.trainloader), 'Loss: %.2f'
                                 % (test_loss/(batch_idx+1)))

        if verbose:
            print('Finished Training')

    def test(self, net, verbose):
        """Distributed network evaluation.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network.

        verbose : bool
            If True, a progress bar will be displayed.

        Returns
        -------
        test_accuracy : float
            The average accuracy across all workers.

        """
        test_accuracy = 0
        net.eval()
        net.to(device)
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                if hvd.rank() == 0 and verbose:
                    progress_bar(batch_idx, len(self.testloader),
                                 'Loss: %.2f | Acc: %.2f%%'
                                 % (test_loss/(batch_idx+1),
                                    100 * correct/total))

        # print(hvd.rank())
        test_loss /= len(self.testsampler)
        test_accuracy = 100 * correct / total

        test_loss = MPI.COMM_WORLD.allreduce(test_loss)/hvd.size()
        test_accuracy = MPI.COMM_WORLD.allreduce(test_accuracy)/hvd.size()

        if hvd.rank() == 0 and verbose:

            print('Accuracy of the network on the test images: %.2f %%' % (
                test_accuracy))
            print('Loss of the network on the test images: %.4f ' % (
                test_loss))
        return test_accuracy / 100.0

    def descriptor_evaluate(self, descriptor, untrained=False, verbose=False):
        """Distributed network evaluation, with a NeuralDescriptor input.

        Parameters
        ----------
        descriptor : NeuralDescriptor
            The neural network's descriptor object.

        untrained : bool (optional)
            If True, skip the training.

        verbose : bool (optional)
            If True, skip the prints.

        Returns
        -------
        test_accuracy : float
            The average accuracy across all workers.

        """

        self.__load_data__()
        descriptor = MPI.COMM_WORLD.bcast(descriptor)

        sample = torch.Tensor(
            self.get_sample()).transpose(2, 0)
        sample = sample.unsqueeze(0)
        net = None
        try:
            net = NeuralNet(descriptor, 10, sample)
        except Exception as identifier:
            if verbose:
                raise(identifier)
            return 0.0

        hvd.broadcast_parameters(net.state_dict(), root_rank=0)
        if not untrained:
            self.train(net, EPOCHS, verbose)
        rets = 0
        rets = self.test(net, verbose)
        if hvd.rank() > 0:
            rets = 0

        hvd.broadcast_parameters(net.state_dict(), root_rank=0)

        return -rets

    def get_sample(self):
        """Returns a sample of the dataset.
        """
        return self.trainloader.dataset.train_data[0]
