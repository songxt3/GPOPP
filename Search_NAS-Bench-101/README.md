# GPOPP on Nas-Bench-101

The codes have been tested on Python 3.6.

Dependent packages:

- nasbench (see  https://github.com/google-research/nasbench)
- tensorflow (==1.15.0)
- scikit-learn
- matplotlib
- scipy

Please download the NAS-Bech-101 subset of the dataset with only models trained at 108 epochs: https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord. (More details are in https://github.com/google-research/nasbench, and you may be required to install additional dependencies like TensorFlow.) Then put the file *nasbench_only108.tfrecord* under the *path* folder.

*Please note that:* if you want to test on NPNAS, please follow the README.md in the subfolder *GCNpredictor*.

## How to run

### GPOPP

```
python GAon101/GP_Surrogate_evolve.py
```

You can change the ac_function, first_archive_size and population_size to see the results in ablation study.

### Offline e2epp

```
python GAon101/offline_evolve.py --predictor 'e2epp'
```

### Offline GCN

```
python GAon101/offline_evolve.py --predictor 'GCN'
```

### Offline GP

```
python GAon101/GP_offline.py
```

