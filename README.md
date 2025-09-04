# Online Performance Predictor for Evolutionary Neural Architecture Search

This is the code implementation for the paper "Online Performance Predictor for Evolutionary Neural Architecture Search".

The experiments on GPOPP can be classified into three categories: the search on  in DARTS serach space, transferring the searched architecture to ImageNet and the search on Nas-Bench-101 search space.

The structure of the project:
```
GPOPP/
├── Search_DARTS_Classification # The search on CIFAR10 in DARTS serach space and test on CIFAR10, ImageNet32, and ImageNet 
├── Search_NAS-Bench-101 # The search on Nas-Bench-101 search space
└── Search_DARTS_MRI # The search on brain MRI in DARTS-like serach space
```

**You can follow the README.md in each subfolder to run the code.**

*The checkpoint of the ImagetNet32 network model is available at: https://drive.google.com/file/d/1PEs4lxRQIJXszCxb3Lt16_-JSZFeRWc7/view?usp=drive_link*

## Reference
```
@ARTICLE{11145879,
  author={Song, Xiaotian and Ji, Han and Zhang, Jian and Liu, Jiyuan and Gao, Shangce and Sun, Yanan},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems}, 
  title={Online Performance Predictor for Evolutionary Neural Architecture Search}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TSMC.2025.3599175}}
```
