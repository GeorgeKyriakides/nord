"""
Example usage of LocalEvaluator,
which evaluates the given architecture on various datasets.

"""
import torch.nn as nn
import torch.optim as opt

from nord.neural_nets import LocalEvaluator, NeuralDescriptor
from nord.utils import assure_reproducibility

from nord.configurations.all import Configs

# Try to make results reproducible
assure_reproducibility()

# Instantiate the evaluator
# define the optimizer CLASS and
# any parameters you wish to utilize
# with the optimizer

evaluator = LocalEvaluator(optimizer_class=opt.Adam,
                           optimizer_params={}, verbose=True)

# Select dataset from:
# - cifar10 (requires 2d conv and pool, i.e. conv = nn.Conv2d, pool = nn.MaxPool2d)
# - fashion-mnist (requires 2d conv and pool, i.e. conv = nn.Conv2d, pool = nn.MaxPool2d)
# - activity_recognition (requires 1d conv and pool, i.e. conv = nn.Conv1d, pool = nn.MaxPool1d)
dataset = 'activity_recognition'

# Instantiate a descriptor
d = NeuralDescriptor()
conf = Configs()

# Define layer presets
conv = nn.Conv1d
# Note that the first layer has number of channels
# equal to the dataset
# This can be avoided if we define an nn.Identity as the
# first layer, before the first Conv1d layer
conv_params = {'in_channels': conf.CHANNELS[dataset],
               'out_channels': 5, 'kernel_size': 3}

conv_2_params = {'in_channels': 5,
                 'out_channels': 10, 'kernel_size': 3}

pool = nn.MaxPool1d
pool_params = {'kernel_size': 2, 'stride': 2}

# Add Torch layers sequentially

d.add_layer(conv, conv_params)
d.add_layer_sequential(conv, conv_2_params)
d.add_layer_sequential(pool, pool_params)


# Evaluate the network for 2 epochs
loss, metrics, total_time = evaluator.descriptor_evaluate(
    descriptor=d, epochs=2, dataset=dataset)

# Print the results

print('Train time: %.2f' % total_time)
print('Test metrics: ', [(key+': %.2f' % value)
                         for key, value in metrics.items()])
print('Test loss: %.2f' % loss)
