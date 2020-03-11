import traceback

import numpy as np
import torch
import torch.nn
import torch.optim

from architectural_design.metaheuristics.evolutionary.arc import Genome

from configs import CHANNELS, INPUT_SHAPE, NUM_CLASSES, PROBLEM_TYPE
from neural_nets import LocalEvaluator, NeuralDescriptor, NeuralNet
from utils import count_parameters

dataset = 'fashion-mnist'

torch.manual_seed(123456)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(123456)
g = Genome.__from_repr__(

    "{'Connections': {1028: {'I': 1028, 'V': (-2, 9999), 'E': False},\
    1030: {'I': 1030, 'V': (-2, 1029), 'E': False}, \
    1031: {'I': 1031, 'V': (1029, 9999), 'E': False},\
    1033: {'I': 1033, 'V': (1029, 1032), 'E': True},\
    1034: {'I': 1034, 'V': (1032, 9999), 'E': True},\
    1036: {'I': 1036, 'V': (-2, 1035), 'E': False},\
    1037: {'I': 1037, 'V': (1035, 1029), 'E': False},\
    1039: {'I': 1039, 'V': (1035, 1038), 'E': True},\
    1040: {'I': 1040, 'V': (1038, 1029), 'E': False},\
    1302: {'I': 1302, 'V': (1038, 1301), 'E': True},\
    1303: {'I': 1303, 'V': (1301, 1029), 'E': True},\
    1305: {'I': 1305, 'V': (-2, 1304), 'E': True},\
    1306: {'I': 1306, 'V': (1304, 1035), 'E': True},\
    1308: {'I': 1308, 'V': (1029, 1307), 'E': True},\
    1309: {'I': 1309, 'V': (1307, 9999), 'E': True}},\
    'Nodes': {-2: {'I': -2, 'V': 'IO', 'E': True},\
    9999: {'I': 9999, 'V': 'IO', 'E': True},\
    1029: {'I': 1029, 'V': [5], 'E': True},\
    1032: {'I': 1032, 'V': [1], 'E': True},\
    1035: {'I': 1035, 'V': [6], 'E': True},\
    1038: {'I': 1038, 'V': [7], 'E': True},\
    1301: {'I': 1301, 'V': [8], 'E': True},\
    1304: {'I': 1304, 'V': [6], 'E': True},\
    1307: {'I': 1307, 'V': [1], 'E': True}}}")
d = g.to_descriptor()

s = d.__repr__()
nd = NeuralDescriptor.__from_repr__(s)

nn = NeuralNet(
    d, NUM_CLASSES[dataset],
    INPUT_SHAPE[dataset],
    CHANNELS[dataset],
    untrained=False,
    problem_type=PROBLEM_TYPE[dataset])

nn.load_state_dict(torch.load('best_architecture_state9446.dict'))
nn.eval()
print(nn)
print(count_parameters(nn))


ne = LocalEvaluator(torch.optim.Adamax, {'weight_decay': 1e-5}, True)


# Evaluation

try:
    fitness = ne.net_evaluate(
        nn, 0, data_percentage=1, dataset=dataset)
    torch.cuda.empty_cache()

except Exception as e:
    print('INVALID')
    print(d)
    print(str(e))
    traceback.print_exc()

print(fitness)
