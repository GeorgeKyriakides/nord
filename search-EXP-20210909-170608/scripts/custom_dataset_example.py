"""
    Example of implementing a custom dataset
"""
import torchvision
import torch
from torchvision import transforms
from nord.configurations.all import Configs
from nord.neural_nets import LocalEvaluator, NeuralDescriptor


# Percentage dictates what percentage of the trainset
# will be used while training. This is not implemented in
# this toy example.
def get_mnist(percentage: float = 1):
    print('Loading MNIST.')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./my_data/mnist',
                                          train=True,
                                          download=True,
                                          transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              num_workers=0,
                                              shuffle=True)

    testset = torchvision.datasets.MNIST(root='./my_data/mnist',
                                         train=False,
                                         download=True,
                                         transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=0)

    classes = ('0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9')

    return trainloader, testloader, classes


# Instantiate the Configs singleton and add the dataset
conf = Configs()
conf.print_datasets()
conf.add_classification_dataset(name='MNIST', num_classes=10, input_shape=(
    28, 28), channels=1, data_load=get_mnist)
conf.print_datasets()

# Make an small network.
d = NeuralDescriptor()

conv = torch.nn.Conv2d
conv_params = {'in_channels': 1,
               'out_channels': 5, 'kernel_size': 3}

pool = torch.nn.MaxPool2d
pool_params = {'kernel_size': 2, 'stride': 2}


d.add_layer_sequential(conv, conv_params)
d.add_layer_sequential(pool, pool_params)

# Evaluate on our new dataset
evaluator = LocalEvaluator(torch.optim.Adam, {}, verbose=True)
evaluator.descriptor_evaluate(d, epochs=2, dataset='MNIST')
