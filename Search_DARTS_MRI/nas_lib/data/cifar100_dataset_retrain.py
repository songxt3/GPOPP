import torch
import torchvision.transforms as transforms
import numpy as np
import torchvision


def get_cifar100_full_train_loader(root_path, transform=None, batch_size=128):
    if transform is None:
        transform = transforms.Compose(
            [transforms.RandomCrop(size=32, padding=(4, 4, 4, 4)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.50707516, 0.48654887, 0.44091784], std=[0.26733428, 0.25643846, 0.27615047])])
    train_set = torchvision.datasets.CIFAR100(root=root_path, train=True, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=16, pin_memory=True)
    return train_loader


def get_cifar100_full_test_loader(root_path, transform=None, batch_size=128):
    if transform is None:
        transform = transforms.Compose(
            [transforms.RandomCrop(size=32, padding=(4, 4, 4, 4)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.50707516, 0.48654887, 0.44091784], std=[0.26733428, 0.25643846, 0.27615047])])
    test_set = torchvision.datasets.CIFAR100(root=root_path, train=False, download=False, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=16, pin_memory=True)
    return test_loader


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def transforms_cifar100(cutout, cutout_length):
    CIFAR100_MEAN = [0.50707516, 0.48654887, 0.44091784]
    CIFAR100_STD = [0.26733428, 0.25643846, 0.27615047]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    if cutout:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    return train_transform, valid_transform


if __name__ == '__main__':
    print()

