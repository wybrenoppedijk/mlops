import torch
import glob
import numpy as np
import os

def mnist():
    # exchange with the corrupted mnist dataset
    # test = np.load('../../../data/corruptmnist/test.npz', mmap_mode='r')
    x = np.load('../../../data/corruptmnist/test.npz',mmap_mode='r')

    images_test = torch.from_numpy(x.f.images)
    labels_test = torch.from_numpy(x.f.labels)
    # print(test.keys())
    train = []

    for file in glob.glob('../../../data/corruptmnist/train*.npz'):
        x = np.load(file,mmap_mode='r')
        images_train = torch.from_numpy(x.f.images).float()
        labels_train = torch.from_numpy(x.f.labels).float()
        train.append((images_train, labels_train))

    test = (images_test, labels_test)
    return test, train