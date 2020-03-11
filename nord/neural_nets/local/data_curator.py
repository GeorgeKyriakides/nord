"""
Created on 2018-08-05

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import torchvision.transforms as transforms
import torchvision
import torch


def get_cifar10(train_batch=128, test_batch=128):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='neural_nets/data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='neural_nets/data',
                                           train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes
