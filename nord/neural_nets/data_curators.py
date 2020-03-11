
"""
Created on Thu Aug 16 15:14:57 2018

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import math
import numbers
import os.path
import random
import urllib.request
import warnings
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import RandomSampler

from neural_nets.distributed_partial_sampler import DistributedPartialSampler

def erase(img, i, j, h, w, v, inplace=False):
    if not isinstance(img, torch.Tensor):
        raise TypeError('img should be Tensor Image. Got {}'.format(type(img)))

    if not inplace:
        img = img.clone()

    img[:, i:i + h, j:j + w] = v
    return img


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.
    # Examples:
        >>> transform = transforms.Compose([
        >>> transforms.RandomHorizontalFlip(),
        >>> transforms.ToTensor(),
        >>> transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> transforms.RandomErasing(),
        >>> ])
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for attempt in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                if isinstance(value, numbers.Number):
                    v = value
                elif isinstance(value, torch._six.string_classes):
                    v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                elif isinstance(value, (list, tuple)):
                    v = torch.tensor(value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if random.uniform(0, 1) < self.p:
            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value)
            return erase(img, x, y, h, w, v, self.inplace)
        return img

def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


def univariate_data(dataset, lag_window, prediction_window):

    end = len(dataset) - prediction_window

    as_strided = np.lib.stride_tricks.as_strided

    p_data = dataset[:end]
    data = as_strided(p_data, (len(p_data) - (lag_window - 1),
                               lag_window), (p_data.strides * 2))

    data = data.reshape(-1, lag_window, 1)

    t_data = dataset[lag_window:]
    targets = as_strided(t_data, (len(
        t_data) - (prediction_window - 1), prediction_window), (t_data.strides * 2))

    return data, targets


@singleton
class data_storage:

    def __init__(self):
        self.climate_data_loaded = None
        self.climate_data_objects = None

    def __get_climate_data_partial(self, percentage, train_batch, test_batch,
                                   differentiate, lag_window,
                                   prediction_window):
        train_sz = 300000
        root = 'neural_nets/data'
        zip_file_path = root+'/jena_climate_2009_2016.csv.zip'
        csv_file_path = zip_file_path.replace('.zip', '')
        url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'

        if not os.path.isfile(zip_file_path):
            print('Downloading Weather Data.')
            urllib.request.urlretrieve(url, zip_file_path)
            print('Downloaded')

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(root)

        df = pd.read_csv(csv_file_path)
        dataset = df['T (degC)']
        dataset.index = df['Date Time']
        dataset = dataset.diff()
        dataset.dropna(inplace=True)
        dataset = dataset.values

        x_train, y_train = univariate_data(dataset[:train_sz],
                                           lag_window,
                                           prediction_window)

        x_test, y_test = univariate_data(dataset[train_sz:],
                                         lag_window,
                                         prediction_window)

        x_train_tensor = torch.from_numpy(
            x_train).reshape(-1, 1, lag_window).float()
        y_train_tensor = torch.from_numpy(
            y_train).reshape(-1, prediction_window).float()
        x_test_tensor = torch.from_numpy(
            x_test).reshape(-1, 1, lag_window).float()
        y_test_tensor = torch.from_numpy(
            y_test).reshape(-1, prediction_window).float()

        if torch.cuda.is_available():
            x_train_tensor, y_train_tensor = x_train_tensor.cuda(), y_train_tensor.cuda()
            x_test_tensor, y_test_tensor = x_test_tensor.cuda(), y_test_tensor.cuda()

        trainset = torch.utils.data.TensorDataset(
            x_train_tensor, y_train_tensor)

        trainsampler = DistributedPartialSampler(
            trainset, percentage, num_replicas=1, rank=0)

        # We want to retain the temporal structure of our dataset.
        # Thus, we do not shuffle!
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                                  num_workers=0,
                                                  sampler=trainsampler,
                                                  shuffle=False)

        testset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                                 shuffle=False, num_workers=0)

        classes = ['forward_'+str(i) for i in range(1, prediction_window+1)]

        return trainloader, testloader, classes

    def get_climate_data_partial(self, percentage, train_batch, test_batch,
                                 differentiate, lag_window,
                                 prediction_window):

        if not self.climate_data_loaded == (percentage, train_batch, test_batch,
                                            differentiate, lag_window,
                                            prediction_window):
            print('GENERATING DATASET')
            self.climate_data_loaded = (percentage, train_batch, test_batch,
                                        differentiate, lag_window,
                                        prediction_window)

            self.climate_data_objects = self.__get_climate_data_partial(percentage, train_batch, test_batch,
                                                                        differentiate, lag_window,
                                                                        prediction_window)

        return self.climate_data_objects


def get_fashion_mnist_partial(percentage=1, train_batch=128, test_batch=128):

    #ORIGINAL norm = transforms.Normalize((0.1307,), (0.3081,))
    #COMPUTED and divided with 255 norm = transforms.Normalize((0.286,), (0.353,))
    #NOT DIVIDED:
    norm = transforms.Normalize((0.1307,), (0.3081,))
    transform = transforms.Compose(
    [
        # RandomCrop(28, padding=4) added after ADAM_WD1e-5 experiment
        # transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
        # value=0.4914 added after ADAM_WD1e-5 experiment
        RandomErasing()
    ])

    test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        norm
    ])

    trainset = torchvision.datasets.FashionMNIST(root='neural_nets/data',
                                                    train=True,
                                                    download=True,
                                                    transform=transform)

    trainsampler = DistributedPartialSampler(
        trainset, percentage, num_replicas=1, rank=0)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                                num_workers=0, sampler=trainsampler)

    testset = torchvision.datasets.FashionMNIST(root='neural_nets/data',
                                                train=False,
                                                download=True,
                                                transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                                shuffle=False, num_workers=0)

    classes = ('t-shirt', 'trouser', 'pullover', 'dress',
                'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')

    return trainloader, testloader, classes


def get_cifar10_distributed(size, rank, train_batch=256, test_batch=256):
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
    n_workers = 4
    train_batch = int(train_batch/n_workers)
    test_batch = int(test_batch/n_workers)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='neural_nets/data',
                                            train=True,
                                            download=False,
                                            transform=transform)

    trainsampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=size, rank=rank)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                              sampler=trainsampler,
                                              pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='neural_nets/data',
                                           train=False,
                                           download=False,
                                           transform=transform)

    testsampler = torch.utils.data.distributed.DistributedSampler(
        testset, num_replicas=size, rank=rank)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             sampler=testsampler,
                                             pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, trainsampler, testloader, testsampler, classes

def get_cifar10_distributed_partial(size, rank, percentage,
                                    train_batch=256, test_batch=256):

    n_workers = 4
    train_batch = int(train_batch/n_workers)
    test_batch = int(test_batch/n_workers)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='neural_nets/data',
                                            train=True,
                                            download=False,
                                            transform=transform)

    trainsampler = DistributedPartialSampler(
        trainset, percentage, num_replicas=size, rank=rank)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                              sampler=trainsampler,
                                              pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='neural_nets/data',
                                           train=False,
                                           download=False,
                                           transform=transform)
    testsampler = torch.utils.data.distributed.DistributedSampler(
        testset, num_replicas=size, rank=rank)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             sampler=testsampler,
                                             pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, trainsampler, testloader, testsampler, classes


def get_cifar10(train_batch=256, test_batch=256):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='neural_nets/data',
                                            train=True,
                                            download=False,
                                            transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='neural_nets/data',
                                           train=False,
                                           download=False,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def get_cifar10_partial(percentage, train_batch=256, test_batch=256):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='neural_nets/data',
                                            train=True,
                                            download=False,
                                            transform=transform)

    trainsampler = DistributedPartialSampler(
        trainset, percentage, num_replicas=1, rank=0)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                              num_workers=0,
                                              sampler=trainsampler)

    testset = torchvision.datasets.CIFAR10(root='neural_nets/data',
                                           train=False,
                                           download=False,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def get_climate_data(train_batch=128, test_batch=128,
                     differentiate=True, lag_window=6*24*5,
                     prediction_window=6):

    train_sz = 300000
    root = 'neural_nets/data'
    zip_file_path = root+'/jena_climate_2009_2016.csv.zip'
    csv_file_path = zip_file_path.replace('.zip', '')
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'

    if not os.path.isfile(zip_file_path):
        print('Downloading Weather Data.')
        urllib.request.urlretrieve(url, zip_file_path)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(root)

    df = pd.read_csv(csv_file_path)
    dataset = df['T (degC)']
    dataset.index = df['Date Time']
    dataset = dataset.diff()
    dataset.dropna(inplace=True)
    dataset = dataset.values

    x_train, y_train = univariate_data(dataset[:train_sz],
                                       lag_window,
                                       prediction_window)

    x_test, y_test = univariate_data(dataset[train_sz:],
                                     lag_window,
                                     prediction_window)

    x_train_tensor = torch.from_numpy(
        x_train).reshape(-1, 1, lag_window).float()
    y_train_tensor = torch.from_numpy(
        y_train).reshape(-1, prediction_window).float()
    x_test_tensor = torch.from_numpy(x_test).reshape(-1, 1, lag_window).float()
    y_test_tensor = torch.from_numpy(
        y_test).reshape(-1, prediction_window).float()

    trainset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)

    # We want to retain the temporal structure of our dataset.
    # Thus, we do not shuffle!
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                              shuffle=False, num_workers=0)

    testset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch,
                                             shuffle=False, num_workers=0)

    classes = ['forward_'+str(i) for i in range(1, prediction_window+1)]

    return trainloader, testloader, classes


def get_climate_data_partial(percentage, train_batch=128, test_batch=128,
                             differentiate=True, lag_window=6*24*5,
                             prediction_window=1):

    storage = data_storage()
    return storage.get_climate_data_partial(percentage, train_batch, test_batch,
                                            differentiate, lag_window,
                                            prediction_window)
