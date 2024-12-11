# GPOPP in DARTS search space for the image classification task

The codes have been tested on Python 3.7.

Dependent packages:

- pytorch
- sklearn
- scipy

And please download the CIFAR-10, ImageNet32, and ImageNet, and set the right path of these dataset.

## How to run

### Search Stage

```
python train_multiple_gpus_open_domain.py
```

### Test Stage

CIFAR10:
```
python tools_open_domain/train_darts_cifar10.py --save_dir [your save dir]
```

ImageNet32:
```
python tools_open_domain/train_darts_imgnet32.py --save_dir [your save dir]
```

ImageNet:
```
python tools_open_domain/Train_ImageNet/train_imagenet.py
```
