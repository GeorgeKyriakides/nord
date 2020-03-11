"""
Created on 2018-08-06

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import torchvision.transforms as transforms
import torchvision
import torch
import torch.utils.data.distributed


def get_cifar10_distributed(size, rank, train_batch=128, test_batch=128):
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

    trainloader : DistributedSampler
        A data sampler for distributed testing.

    classes : tuple
        A tuple with the classes' names.

    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='neural_nets/data-%d' % rank,
                                            train=True,
                                            download=True,
                                            transform=transform)

    trainsampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=size, rank=rank)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                              sampler=trainsampler)

    testset = torchvision.datasets.CIFAR10(root='neural_nets/data-%d' % rank,
                                           train=False,
                                           download=True,
                                           transform=transform)
    testsampler = torch.utils.data.distributed.DistributedSampler(
        testset, num_replicas=size, rank=rank)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             sampler=testsampler)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, trainsampler, testloader, testsampler, classes
