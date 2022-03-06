import torch.nn as nn

from nord.neural_nets import NaswotEvaluator, NeuralDescriptor
from nord.utils import assure_reproducibility

from nord.configurations.all import Configs


# Try to make results reproducible
# assure_reproducibility()

# Instantiate the evaluator
naswot_evaluator = NaswotEvaluator()

# Select dataset from:
# - cifar10 (batch_size=256)
# - fashion-mnist (batch_size=128)
dataset = 'cifar10'
batch_size = 1024

# Instantiate a descriptor
d = NeuralDescriptor()
conf = Configs()

# Define layer presets
conv = nn.Conv2d
# Note that the first layer has number of channels
# equal to the dataset
# This can be avoided if we define an nn.Identity as the
# first layer, before the first Conv1d layer
conv_params = {'in_channels': conf.CHANNELS[dataset],
               'out_channels': 5, 'kernel_size': 3}

conv_2_params = {'in_channels': 5,
                 'out_channels': 10, 'kernel_size': 3}

pool = nn.MaxPool2d
pool_params = {'kernel_size': 2, 'stride': 2}

# Add Torch layers sequentially
d.add_layer(conv, conv_params)
d.add_layer_sequential(conv, conv_2_params)
d.add_layer_sequential(pool, pool_params)

print(d)

K, score = naswot_evaluator.descriptor_evaluate(descriptor=d, batch_size=batch_size, data_percentage=1.0,
                                                untrained=True, dataset=dataset)

print('K:', K)
print('score:', score)
