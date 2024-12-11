from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.vision import VisionDataset


class Imagenet32(VisionDataset):
    base_folder = 'imagenet32-batches-py'
    train_list = [
        'train_data_batch_1',
        'train_data_batch_2',
        'train_data_batch_3',
        'train_data_batch_4',
        'train_data_batch_5',
        'train_data_batch_6',
        'train_data_batch_7',
        'train_data_batch_8',
        'train_data_batch_9',
        'train_data_batch_10',
    ]

    test_list = [
        'val_data',
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Imagenet32, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # please note that the original targets are in [1, 1000] which dose not conform with nn.CrossEntropyLoss
        # and the targets should -1
        img, target = self.data[index], self.targets[index] - 1

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


