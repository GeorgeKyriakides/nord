
import torch.nn as nn
from nord.data_curators import (get_activity_recognition_data, get_cifar10,
                                get_fashion_mnist, get_fmnist_graphs,
                                get_nasbench_graph_data_loaders)
from nord.losses import nord_cross_entropy_loss, nord_margin_rank_loss
from nord.metrics import MSE, accuracy
from nord.metrics import binary_rank_correlation_tau_with_top as br
from nord.metrics import one_hot_accuracy
from nord.metrics import regression_rank_correlation_with_top as rr
from nord.utils import singleton

from .dense_parts import get_dense_batch_net, get_dense_net, get_graph_rank_net


@singleton
class Configs:
    def __init__(self):
        self.NUM_CLASSES = {'cifar10': 10,
                            'activity_recognition': 7,
                            'fashion-mnist': 10,
                            'graph-nasbench3': 3,
                            'graph-nasbench1': 1,
                            'graph-f-mnist3': 3,
                            'graph-f-mnist1': 1}

        self.INPUT_SHAPE = {'cifar10': (32, 32),
                            'activity_recognition': (52,),
                            'fashion-mnist': (28, 28),
                            'graph-nasbench3': (5, ),
                            'graph-nasbench1': (5, ),
                            'graph-f-mnist3': (5, ),
                            'graph-f-mnist1': (5, )}

        self.CHANNELS = {'cifar10': 3,
                         'activity_recognition': 3,
                         'fashion-mnist': 1,
                         'graph-nasbench3': 5,
                         'graph-nasbench1': 5,
                         'graph-f-mnist3': 5,
                         'graph-f-mnist1': 5}

        self.CRITERION = {'cifar10': nn.CrossEntropyLoss,
                          'activity_recognition': nord_cross_entropy_loss,
                          'fashion-mnist': nn.CrossEntropyLoss,
                          'graph-nasbench3': nn.KLDivLoss,
                          'graph-nasbench1': nord_margin_rank_loss,
                          'graph-f-mnist3': nn.KLDivLoss,
                          'graph-f-mnist1': nord_margin_rank_loss}

        self.METRIC = {'cifar10':  [accuracy],
                       'activity_recognition': [one_hot_accuracy],
                       'fashion-mnist':  [accuracy],
                       'graph-nasbench3': [br(0.5), br(0.5, 'spearman')],
                       'graph-nasbench1': [rr(0.5), rr(0.5, 'spearman')],
                       'graph-f-mnist3': [one_hot_accuracy, br(0.5), br(0.5, 'spearman')],
                       'graph-f-mnist1': [MSE, rr(0.5), rr(0.5, 'spearman')]}

        # Forces hidden layer outputs to certain dimensions
        self.DIMENSION_KEEPING = {'cifar10': None,
                                  'fashion-mnist': None,
                                  'activity_recognition': None,
                                  'graph-nasbench3': None,
                                  'graph-nasbench1': None,
                                  'graph-f-mnist3': None,
                                  'graph-f-mnist1': None}

        graph_bs = 256

        def nasbench_classification(data_perc):
            return get_nasbench_graph_data_loaders(graph_bs,
                                                   initial_pop=20, mutated=980, classes=3, val_perc=0.0,
                                                   test_perc=0.3, all_vs_all=[False, True])

        def nasbench_regression(data_perc):
            return get_nasbench_graph_data_loaders(graph_bs,
                                                   initial_pop=20, mutated=980, classes=1, val_perc=0.0,
                                                   test_perc=0.3, all_vs_all=[False, True])

        def fmnist_classification(data_perc):
            return get_fmnist_graphs(graph_bs, val_perc=0.0,
                                     test_perc=0.3, classes=3, all_vs_all=[False, True])

        def fmnist_regression(data_perc):
            return get_fmnist_graphs(batch_size=graph_bs, val_perc=0.0,
                                     test_perc=0.3, classes=1, all_vs_all=[False, True])

        self.DATA_LOAD = {'cifar10': get_cifar10,
                          'fashion-mnist': get_fashion_mnist,
                          'activity_recognition': get_activity_recognition_data,
                          'graph-nasbench3': nasbench_classification,
                          'graph-nasbench1': nasbench_regression,
                          'graph-f-mnist3': fmnist_classification,
                          'graph-f-mnist1': fmnist_regression}

        self.DENSE_PART = {'cifar10': get_dense_batch_net(3, self.NUM_CLASSES['cifar10'], nn.LogSoftmax(dim=1)),
                           'fashion-mnist': get_dense_net(2, self.NUM_CLASSES['fashion-mnist'], nn.LogSoftmax(dim=1)),
                           'activity_recognition': get_dense_net(3, self.NUM_CLASSES['activity_recognition'], nn.LogSoftmax(dim=1)),
                           'graph-nasbench3': get_graph_rank_net(2, self.NUM_CLASSES['graph-nasbench3'], nn.LogSoftmax(dim=1)),
                           'graph-nasbench1': get_graph_rank_net(2, self.NUM_CLASSES['graph-nasbench1'], nn.Softplus()),
                           'graph-f-mnist3': get_graph_rank_net(2, self.NUM_CLASSES['graph-nasbench3'], nn.LogSoftmax(dim=1)),
                           'graph-f-mnist1': get_graph_rank_net(2, self.NUM_CLASSES['graph-nasbench1'], nn.Softplus())
                           }

        self.INPUT_DATA_ORGANIZATION = {'cifar10': 'matrix',
                                        'activity_recognition': 'matrix',
                                        'fashion-mnist': 'matrix',
                                        'graph-nasbench3': 'graph',
                                        'graph-nasbench1': 'graph',
                                        'graph-f-mnist3': 'graph',
                                        'graph-f-mnist1': 'graph'}

    def print_datasets(self):
        print(list(self.DENSE_PART.keys()))

    def add_dataset(self, name: str, num_classes: int,
                    input_shape: tuple, channels: int, criterion: object,
                    metrics: tuple, dimension_keeping: int, data_load: object, dense_part: object, data_organization: str):

        if data_organization not in ['matrix', 'graph']:
            raise Exception(
                "data_organization should be either 'graph' for graph data or 'matrix' for vectorized data (i.e. time series, images)")

        self.NUM_CLASSES[name] = num_classes
        self.INPUT_SHAPE[name] = input_shape
        self.CHANNELS[name] = channels
        self.CRITERION[name] = criterion
        self.METRIC[name] = metrics
        self.DIMENSION_KEEPING[name] = dimension_keeping
        self.DATA_LOAD[name] = data_load
        self.DENSE_PART[name] = dense_part
        self.INPUT_DATA_ORGANIZATION[name] = data_organization

    def add_classification_dataset(self, name: str, num_classes: int,
                                   input_shape: tuple, channels: int, data_load, data_organization='matrix'):
        self.add_dataset(name, num_classes, input_shape, channels,
                         nn.CrossEntropyLoss, [accuracy], None, data_load, get_dense_batch_net(
                             3, num_classes, nn.LogSoftmax(dim=1)), data_organization)

    def add_regression_dataset(self, name: str, num_targets: int,
                               input_shape: tuple, channels: int, data_load, data_organization='matrix'):
        self.add_dataset(name, num_targets, input_shape, channels,
                         nn.CrossEntropyLoss, [MSE], None, data_load, get_dense_batch_net(
                             3, num_targets, nn.LogSoftmax(dim=1)), data_organization)
