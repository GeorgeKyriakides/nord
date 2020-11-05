import torch.nn as nn
from nord import losses, metrics
from nord.data_curators import (get_activity_recognition_data, get_cifar10,
                                get_fashion_mnist)
from nord.utils import singleton

from .dense_parts import get_dense_batch_net, get_dense_net


@singleton
class Configs:
    def __init__(self):
        self.NUM_CLASSES = {'cifar10': 10,
                            'activity_recognition': 7,
                            'fashion-mnist': 10}

        self.INPUT_SHAPE = {'cifar10': (32, 32),
                            'activity_recognition': (52,),
                            'fashion-mnist': (28, 28)}

        self.CHANNELS = {'cifar10': 3,
                         'activity_recognition': 3,
                         'fashion-mnist': 1}

        self.CRITERION = {'cifar10': nn.CrossEntropyLoss,
                          'activity_recognition': losses.My_CrossEntropyLoss,
                          'fashion-mnist': nn.CrossEntropyLoss}

        self.METRIC = {'cifar10':  [metrics.accuracy],
                       'activity_recognition': [metrics.one_hot_accuracy],
                       'fashion-mnist':  [metrics.accuracy]}

        # Forces hidden layer outputs to certain dimensions
        self.DIMENSION_KEEPING = {'cifar10': None,
                                  'fashion-mnist': None,
                                  'activity_recognition': None}

        self.DATA_LOAD = {'cifar10': get_cifar10,
                          'fashion-mnist': get_fashion_mnist,
                          'activity_recognition': get_activity_recognition_data}

        self.DENSE_PART = {'cifar10': get_dense_batch_net(3, self.NUM_CLASSES['cifar10'], nn.LogSoftmax(dim=1)),
                           'fashion-mnist': get_dense_net(2, self.NUM_CLASSES['fashion-mnist'], nn.LogSoftmax(dim=1)),
                           'activity_recognition': get_dense_net(3, self.NUM_CLASSES['activity_recognition'], nn.LogSoftmax(dim=1)),
                           }

    def print_datasets(self):
        print(list(self.DENSE_PART.keys()))

    def add_dataset(self, name: str, num_classes: int,
                    input_shape: tuple, channels: int, criterion: object,
                    metrics: tuple, dimension_keeping: int, data_load: object, dense_part: object):

        self.NUM_CLASSES[name] = num_classes
        self.INPUT_SHAPE[name] = input_shape
        self.CHANNELS[name] = channels
        self.CRITERION[name] = criterion
        self.METRIC[name] = metrics
        self.DIMENSION_KEEPING[name] = dimension_keeping
        self.DATA_LOAD[name] = data_load
        self.DENSE_PART[name] = dense_part

    def add_classification_dataset(self, name: str, num_classes: int,
                                   input_shape: tuple, channels: int, data_load):
        self.add_dataset(name, num_classes, input_shape, channels,
                         nn.CrossEntropyLoss, [metrics.accuracy], None, data_load, get_dense_batch_net(
                             3, self.NUM_CLASSES['cifar10'], nn.LogSoftmax(dim=1)))

    def add_regression_dataset(self, name: str, num_classes: int,
                               input_shape: tuple, channels: int, data_load):
        self.add_dataset(name, num_classes, input_shape, channels,
                         nn.CrossEntropyLoss, [metrics.accuracy], None, data_load, get_dense_batch_net(
                             3, self.NUM_CLASSES['cifar10'], nn.LogSoftmax(dim=1)))
