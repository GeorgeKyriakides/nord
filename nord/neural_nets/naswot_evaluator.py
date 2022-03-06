import numpy as np
import torch
from .neural_descriptor import NeuralDescriptor
from .neural_builder import NeuralNet
from nord.configurations.all import Configs
import time


class NaswotEvaluator():
    """A class to load a dataset and evaluate a network based on the naswot metric
    """

    def __init__(self):
        self.dataset = None
        (self.trainloader, self.trainsampler,
         self.testloader, self.testsampler,
         self.classes) = [None] * 5
        self.data_loaded = False
        self.conf = Configs()

    def load_data(self, data_percentage: float, dataset: str, batch_size: int):

        if not self.data_loaded:
            self.data_loaded = True
            self.dataset = dataset

            (self.trainloader,
             self.testloader,
             self.classes) = self.conf.DATA_LOAD[self.dataset](data_percentage, train_batch=batch_size, test_batch=batch_size)

    def descriptor_to_net(self, descriptor: NeuralDescriptor, untrained: bool):
        """Make a net from descriptor, accounting for any distributed comms
        """
        return NeuralNet(descriptor, self.conf.NUM_CLASSES[self.dataset],
                         self.conf.INPUT_SHAPE[self.dataset], self.conf.CHANNELS[self.dataset],
                         untrained=untrained,
                         keep_dimensions=self.conf.DIMENSION_KEEPING[self.dataset],
                         dense_part=self.conf.DENSE_PART[self.dataset])

    def descriptor_evaluate(self, descriptor: NeuralDescriptor,
                            batch_size: int = 256, data_percentage: float = 1.0,
                            untrained: bool = True, dataset: str = None):
        """Distributed network evaluation, with a NeuralDescriptor input.

        Parameters
        ----------
        descriptor : NeuralDescriptor
            The neural network's descriptor object.

        untrained : bool (optional)
            If True, skip the training.

        Returns
        -------
        K : numpy.ndarray
            The kernel matrix.

        score: numpy.float64
            The naswot score.

        """
        net = self.descriptor_to_net(descriptor, untrained)
        print(net)

        return self.net_evaluate(net, batch_size, data_percentage, untrained, dataset)

    def net_evaluate(self, net, batch_size: int = 256, data_percentage: float = 1.0,
                     untrained: bool = True, dataset: str = None):

        with torch.no_grad():

            self.load_data(data_percentage, dataset, batch_size)

            # this is K
            def counting_forward_hook(module, inp, out):
                inp = inp[0].view(inp[0].size(0), -1)
                x = (inp > 0).float()  # binary indicator
                K = x @ x.t()
                K2 = (1. - x) @ (1. - x.t())
                net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()  # hamming distance

            def counting_backward_hook(module, inp, out):
                module.visited_backwards = True

            net.K = np.zeros((batch_size, batch_size))

            for name, module in net.named_modules():
                if 'ReLU' in str(type(module)):
                    module.register_forward_hook(counting_forward_hook)
                    module.register_full_backward_hook(counting_backward_hook)

            # run batch through network
            x, target = next(iter(self.trainloader))

            x2 = torch.clone(x).cuda()

            net(x2)

            # this is the logarithm of the determinant of K

            def hooklogdet(K, labels=None):
                s, ld = np.linalg.slogdet(K)
                return ld

            return net.K, hooklogdet(net.K, target)
