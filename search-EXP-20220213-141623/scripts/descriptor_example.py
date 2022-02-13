"""
Example usage of NeuralDescriptor and NeuralNet classes

"""
import torch.nn as nn

from nord.neural_nets import NeuralDescriptor, NeuralNet

# Define layer presets
conv = nn.Conv2d
conv_params = {'in_channels': 3,
               'out_channels': 5, 'kernel_size': 3}

conv_2_params = {'in_channels': 5,
                 'out_channels': 10, 'kernel_size': 5}

pool = nn.MaxPool2d
pool_params = {'kernel_size': 2, 'stride': 2}

pool2_params = {'kernel_size': 2, 'stride': 5}

# # Sequential Example
# # Instantiate a descriptor
d = NeuralDescriptor()

# Add layers sequentially
d.add_layer_sequential(conv, conv_params)
d.add_layer_sequential(conv, conv_2_params)
d.add_layer_sequential(pool, pool_params)
d.add_layer_sequential(pool, pool2_params)


# Instantiate the network
net = NeuralNet(net_descriptor=d, num_classes=10,
                input_shape=(32, 32), input_channels=3)
# Print it
print(net)
# Plot it
net.plot()


# Non-Sequential Example
# Re-instantiate the descriptor
d = NeuralDescriptor()

# Add layers and give them names
d.add_layer(conv, conv_params, 'conv')
d.add_layer(conv, conv_2_params, 'conv2')
d.add_layer(conv, {'in_channels': 5,
                   'out_channels': 10, 'kernel_size': 3}, 'conv3')
d.add_layer(pool, pool_params, 'pool1')
d.add_layer(pool, pool2_params, 'pool2')

# Add Connections using names
d.connect_layers('conv', 'conv2')
d.connect_layers('conv2', 'conv3')
d.connect_layers('conv3', 'pool2')
d.connect_layers('conv2', 'pool1')
d.connect_layers('pool1', 'pool2')


# Instantiate the network
net = NeuralNet(net_descriptor=d, num_classes=10,
                input_shape=(128, 128), input_channels=3)
# Print it
print(net)
# Plot it
net.plot()
