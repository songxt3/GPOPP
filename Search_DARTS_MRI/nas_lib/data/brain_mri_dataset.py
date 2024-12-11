import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from skimage import io, transform

csv_path = '/home/xiaotian/dataset/medical/lgg-mri-segmentation/kaggle_3m/data.csv'
data_folder = '/home/xiaotian/dataset/medical/lgg-mri-segmentation/kaggle_3m'
eg_path = '/home/xiaotian/dataset/medical/lgg-mri-segmentation/kaggle_3m/TCGA_HT_8113_19930809'
eg_img = '/home/xiaotian/dataset/medical/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4944_20010208/TCGA_CS_4944_20010208_10.tif'
eg_mask = '/home/xiaotian/dataset/medical/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4944_20010208/TCGA_CS_4944_20010208_10_mask.tif'


class Brain_data(Dataset):
    def __init__(self, path):
        self.path = path
        self.patients = [file for file in os.listdir(path) if file not in ['data.csv', 'README.md']]
        self.masks, self.images = [], []

        for patient in self.patients:
            for file in os.listdir(os.path.join(self.path, patient)):
                if 'mask' in file.split('.')[0].split('_'):
                    self.masks.append(os.path.join(self.path, patient, file))
                else:
                    self.images.append(os.path.join(self.path, patient, file))

        self.images = sorted(self.images)
        self.masks = sorted(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        image = io.imread(image)
        image = transform.resize(image, (256, 256))
        image = image / 255
        image = image.transpose((2, 0, 1))

        mask = io.imread(mask)
        mask = transform.resize(mask, (256, 256))
        mask = mask / 255
        mask = np.expand_dims(mask, axis=-1).transpose((2, 0, 1))

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return (image, mask)

def spilt_brain_mri_train_and_test():
    data = Brain_data(data_folder)
    print('Length of dataset is {}'.format(data.__len__()))
    trainset, valset = random_split(data, [3600, 329], generator=torch.Generator().manual_seed(0))
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=96, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=96, num_workers=2)

    return train_loader, val_loader
