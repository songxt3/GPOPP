import torch
import torchvision.transforms as transforms
import numpy as np
from nas_lib.data.Imagenet32 import Imagenet32


def get_imagenet32_full_train_loader(root_path, transform=None, batch_size=128):
    if transform is None:
        transform = transforms.Compose(
            [transforms.RandomCrop(size=32, padding=(4, 4, 4, 4)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.48109809, 0.45747185, 0.40785507], std=[0.26040889, 0.25321260, 0.26820634])])
    train_set = Imagenet32(root=root_path, train=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)
    return train_loader


def get_imagenet32_full_test_loader(root_path, transform=None, batch_size=128):
    if transform is None:
        transform = transforms.Compose(
            [transforms.RandomCrop(size=32, padding=(4, 4, 4, 4)),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.48109809, 0.45747185, 0.40785507], std=[0.26040889, 0.25321260, 0.26820634])])
    test_set = Imagenet32(root=root_path, train=False, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=4, pin_memory=True)
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


def transforms_imagenet32(cutout, cutout_length):
    IMAGENET32_MEAN = [0.48109809, 0.45747185, 0.40785507]
    IMAGENET32_STD = [0.26040889, 0.25321260, 0.26820634]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET32_MEAN, IMAGENET32_STD),
    ])
    if cutout:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET32_MEAN, IMAGENET32_STD),
    ])
    return train_transform, valid_transform


if __name__ == '__main__':
    print()
