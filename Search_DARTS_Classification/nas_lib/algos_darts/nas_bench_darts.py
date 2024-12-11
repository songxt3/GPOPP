from nas_lib.utils.utils_darts import init_nasbench_macro_cifar10, convert_to_genotype
from hashlib import sha256
from nas_lib.eigen.trainer_nasbench_open_darts_async import async_macro_model_train
from nas_lib.models_darts.darts_graph import nasbench2graph2
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_squared_error
from copy import deepcopy
from scipy.stats import norm
import os
import pickle


def nas_bench_darts_search_open(search_space,
                                 algo_info,
                                 logger,
                                 gpus,
                                 save_dir,
                                 verbose=True,
                                 dataset='cifar10',
                                 seed=111222333):
    GP_macro_graph_dict = {}
    GP_model_keys = []


    # regularized evolution
    num_pop = algo_info['num_pop']

    init_nasbench_macro_cifar10(save_dir)


    # set seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.enabled = True
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

    init_pop_list = search_space.generate_random_dataset(num=num_pop)
    data_dict_keys = [convert_to_genotype(d[0], verbose=False) for d in init_pop_list]
    data_dict_keys = [sha256(str(k).encode('utf-8')).hexdigest() for k in data_dict_keys]
    cur_Pops = {}
    for i, d in enumerate(init_pop_list):
        cur_Pops[data_dict_keys[i]] = list(d)
    for i, d in enumerate(init_pop_list):
        GP_macro_graph_dict[data_dict_keys[i]] = list(d)

    darts_neural_dict = search_space.assemble_cifar10_neural_net(init_pop_list)
    data = async_macro_model_train(model_data=darts_neural_dict,
                                   gpus=gpus,
                                   save_dir=save_dir,
                                   dataset=dataset)
    for k, v in data.items():
        if k not in GP_macro_graph_dict:
            raise ValueError('model trained acc key should in GP_macro_graph_dict')
        GP_macro_graph_dict[k].extend(v)


    for hash_key in GP_macro_graph_dict:
        cur_Pops[hash_key].append(GP_macro_graph_dict[hash_key][2])


    cur_Pops_key_list = []
    for hash_key in cur_Pops:
        cur_Pops[hash_key][1] = None
        cur_Pops_key_list.append(hash_key)

    gen_no = 0
    pop_save_path = save_dir + '/population_pkl/' + str(gen_no) + '.pkl'
    with open(pop_save_path, 'wb') as file:
        pickle.dump(cur_Pops, file)
        pickle.dump(cur_Pops_key_list, file)

    return GP_macro_graph_dict, GP_model_keys
