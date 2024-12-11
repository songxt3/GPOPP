# GPOPP in DARTS search space for the brain MRI task

The codes have been tested on Python 3.6.

Dependent packages:

- pytorch
- sklearn
- scipy

And please download the brain MRI dataset, which is available at https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation.

## How to run

### Search Stage

```
python train_multiple_gpus_open_domain.py
```

### Test Stage

UNet architecture:
```
python tools_open_domain/train_unet.py
```

Searched architecture by GPOPP:
```
python tools_open_domain/train_searched_arch.py
```