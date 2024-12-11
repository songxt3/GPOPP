import os
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def get_imagenet_full_train_loader(root_path, batch_size=128):
    traindir = os.path.join(root_path, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=32,
                                               pin_memory=True)
    return train_loader


def get_imagenet_full_test_loader(root_path, batch_size=128):
    validdir = os.path.join(root_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
                                              shuffle=False, num_workers=32, pin_memory=True)
    return test_loader
