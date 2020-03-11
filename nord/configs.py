import torch.nn as nn


NUM_CLASSES = {'cifar10': 10, 'time_series': 72, 'fashion-mnist': 10}

INPUT_SHAPE = {'cifar10': (32, 32),
               'time_series': (720,),
               'fashion-mnist': (28, 28)}

CHANNELS = {'cifar10': 3, 'time_series': 1, 'fashion-mnist': 1}

CRITERION = {'cifar10': nn.CrossEntropyLoss(),
             'time_series': nn.MSELoss(),
             'fashion-mnist': nn.CrossEntropyLoss()}

PROBLEM_TYPE = {'cifar10': 'classification',
                'time_series': 'regression',
                'fashion-mnist': 'classification'}
