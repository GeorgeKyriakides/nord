"""
Created on 2020-10-10 12:45:00

@author: George Kyriakides
          ge.kyriakides@gmail.com
"""
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from .distributed_partial_sampler import DistributedPartialSampler


def get_cifar10_distributed(size: int, rank: int, train_batch: int = 256,
                            test_batch: int = 256):
    """Distributes the CIFAR10 dataset across workers for distributed training.

    Parameters
    ----------
    size : int
        The total number of workers (as in MPI.COMM_WORLD.size)

    rank : int
        The worker's rank (as in MPI.COMM_WORLD.rank)

    train_batch : int (optional)
        Batch size for training

    test_batch : int (optional)
        Batch size for testing

    Returns
    -------
    trainloader : DataLoader
        A data loader for the training split.

    trainsampler : DistributedSampler
        A data sampler for distributed training.

    testloader : DataLoader
        A data loader for the test split.

    testsampler : DistributedSampler
        A data sampler for distributed testing.

    classes : tuple
        A tuple with the classes' names.

    """
    n_workers = 4
    train_batch = int(train_batch/n_workers)
    test_batch = int(test_batch/n_workers)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10',
                                            train=True,
                                            download=True,
                                            transform=transform)

    trainsampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=size, rank=rank)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                              sampler=trainsampler,
                                              pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10',
                                           train=False,
                                           download=True,
                                           transform=transform)

    testsampler = torch.utils.data.distributed.DistributedSampler(
        testset, num_replicas=size, rank=rank)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             sampler=testsampler,
                                             pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, trainsampler, testloader, testsampler, classes


def get_cifar10(percentage: float = 1, train_batch: int = 256, test_batch: int = 256):
    print('Loading CIFAR-10.')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10',
                                            train=True,
                                            download=True,
                                            transform=transform)

    trainsampler = DistributedPartialSampler(
        trainset, percentage, num_replicas=1, rank=0)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                              num_workers=0,
                                              sampler=trainsampler)

    testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10',
                                           train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def get_fashion_mnist(percentage: float = 1, train_batch: int = 128, test_batch: int = 128):

    print('Loading Fashion MNIST.')
    norm = transforms.Normalize((0.1307,), (0.3081,))
    transform = transforms.Compose(
        [

            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing()
        ])

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            norm
        ])

    trainset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                 train=True,
                                                 download=True,
                                                 transform=transform)

    trainsampler = DistributedPartialSampler(
        trainset, percentage, num_replicas=1, rank=0)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                              num_workers=0, sampler=trainsampler)

    testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                train=False,
                                                download=True,
                                                transform=test_transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             shuffle=False, num_workers=0)

    classes = ('t-shirt', 'trouser', 'pullover', 'dress',
               'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')

    return trainloader, testloader, classes
