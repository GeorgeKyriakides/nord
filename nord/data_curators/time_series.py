import os
import zipfile

import numpy as np
import pandas as pd
import torch
from nord.utils import pdownload
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader, TensorDataset

from .distributed_partial_sampler import DistributedPartialSampler


def get_activity_recognition_data(percentage, train_batch=128, test_batch=128,
                                  differentiate=True, lag_window=52,
                                  lag_overlap_samples=26,
                                  test_subjects=4):

    root = './data/activity_recognition_data'
    zip_file_path = root+'/Activity Recognition from Single Chest-Mounted Accelerometer.zip'
    csv_file_path = zip_file_path.replace('.zip', '')
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00287/Activity%20Recognition%20from%20Single%20Chest-Mounted%20Accelerometer.zip'

    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.isfile(zip_file_path):
        print('Downloading Activity Recognition Data.')
        pdownload(url, zip_file_path)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(root)

    lags = range(0, lag_window)
    lb = LabelBinarizer().fit([x for x in range(1, 8)])

    def get_subjects(files):
        print('Loading Activity Recognition Data.')
        all_data = []
        all_labels = []
        for file in files:
            if file.split('.')[-1] == 'csv':
                df = pd.read_csv(csv_file_path+'/'+file, header=None,
                                 names=['x', 'y', 'z', 'activity'], usecols=[1, 2, 3, 4])
                # Last entry is sometimes 0
                df = df.iloc[:-1]
                if differentiate:
                    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].diff()

                df = df.assign(**{
                    '{} (t-{})'.format(col, t): df[col].shift(t)
                    for col in ['x', 'y', 'z']
                    for t in lags

                })

                df.dropna(inplace=True)
                df = df.iloc[::lag_overlap_samples, :]
                data = df.drop(
                    labels=['activity', 'x', 'y', 'z'], axis=1).values
                labels = lb.transform(df['activity'])
                all_data.extend(np.array(data.reshape(-1, 3, lag_window)))
                all_labels.extend(labels)
        return torch.Tensor(all_data), torch.Tensor(all_labels)

    all_files = os.listdir(csv_file_path)

    trainset = TensorDataset(*get_subjects(all_files[:-test_subjects]))
    testset = TensorDataset(*get_subjects(all_files[-test_subjects:]))

    trainsampler = DistributedPartialSampler(
        trainset, percentage, num_replicas=1, rank=0)

    trainloader = DataLoader(
        trainset, batch_size=train_batch, sampler=trainsampler)
    testloader = DataLoader(testset, batch_size=test_batch, shuffle=True)

    return trainloader, testloader, 7
