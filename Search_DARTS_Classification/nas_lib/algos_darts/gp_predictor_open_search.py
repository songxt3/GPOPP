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


def gp_predictor_search_open(search_space,
                             algo_info,
                             logger,
                             gpus,
                             save_dir,
                             verbose=True,
                             dataset='cifar10',
                             seed=111222333):
    GP_macro_graph_dict = {}
    GP_model_keys = []

    GP_Predictor = GaussianProcessRegressor(kernel=Matern(), normalize_y=True)

    def binary_encoding(train_data):
        train_data_arch_binary_list = []
        for arch in train_data:
            arch1 = []
            arch1.append(arch[0][:15, :15])
            arch1.append(arch[1][:15])
            _, node_f_1 = nasbench2graph2(arch1)

            arch2 = []
            arch2.append(arch[0][15:, 15:])
            arch2.append(arch[1][15:])
            _, node_f_2 = nasbench2graph2(arch2)

            arch_binary_encoding = np.concatenate(
                (arch1[0].reshape(-1), node_f_1.view(-1), arch2[0].reshape(-1), node_f_2.view(-1)))

            train_data_arch_binary_list.append(arch_binary_encoding)

        return train_data_arch_binary_list

    def expected_improvement(x, gaussian_process, evaluated_y, greater_is_better=True, xi=0, n_params=780):
        """ expected_improvement
        Expected improvement acquisition function.
        Arguments:
        ----------
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the expected improvement needs to be computed.
            gaussian_process: GaussianProcessRegressor object.
                Gaussian process trained on previously evaluated hyperparameters.
            evaluated_loss: Numpy array.
                Numpy array that contains the values of the y function for the previously
                evaluated architecture.
            greater_is_better: Boolean.
                Boolean flag that indicates whether the loss function is to be maximised or minimised.
            n_params: int.
                Dimension of the hyperparameter space.
        """

        x_to_predict = np.array(x).reshape(-1, n_params)

        mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

        if greater_is_better:
            loss_optimum = np.max(evaluated_y)
        else:
            loss_optimum = np.min(evaluated_y)

        scaling_factor = (-1) ** (not greater_is_better)

        # In case sigma equals zero
        with np.errstate(divide='ignore'):
            Z = scaling_factor * (mu - loss_optimum) / sigma
            expected_improvement = scaling_factor * (mu - loss_optimum - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            expected_improvement[sigma == 0.0] = 0.0

        return -1 * expected_improvement

    def fit_GP():
        train_data = search_space.assemble_graph(GP_macro_graph_dict, GP_model_keys)
        val_acc = np.array([GP_macro_graph_dict[k][2] for k in GP_model_keys])
        train_data_arch_binary_list = binary_encoding(train_data)
        GP_Predictor.fit(train_data_arch_binary_list, val_acc)

    def pred_by_GP(population_list):
        graph_dict = {}
        data_dict_keys = [convert_to_genotype(d[0], verbose=False) for d in population_list]
        data_dict_keys = [sha256(str(k).encode('utf-8')).hexdigest() for k in data_dict_keys]

        for i, d in enumerate(population_list):
            graph_dict[data_dict_keys[i]] = list(d)

        pred_data = search_space.assemble_graph(graph_dict, data_dict_keys)
        pred_data_arch_binary_list = binary_encoding(pred_data)
        results = GP_Predictor.predict(pred_data_arch_binary_list)
        return results

    def updata_GP(population_list, evaluated_y):
        graph_dict = {}
        data_dict_keys = [convert_to_genotype(d[0], verbose=False) for d in population_list]
        data_dict_keys = [sha256(str(k).encode('utf-8')).hexdigest() for k in data_dict_keys]

        for i, d in enumerate(population_list):
            graph_dict[data_dict_keys[i]] = list(d)

        pred_data = search_space.assemble_graph(graph_dict, data_dict_keys)
        pred_data_arch_binary_list = binary_encoding(pred_data)
        criterion_metric = expected_improvement(pred_data_arch_binary_list, GP_Predictor, evaluated_y)
        sorted_index = np.argsort(criterion_metric)
        best_index = sorted_index[0]
        best_dict_keys = data_dict_keys.pop(best_index)
        best_graph_list = []
        best_graph_list.append(graph_dict.pop(best_dict_keys))
        copy_population_list = deepcopy(population_list)
        copy_population_list.pop(best_index)
        darts_neural_dict = search_space.assemble_cifar10_neural_net(best_graph_list)
        # test = True
        # if test:
        #     for hash_key in darts_neural_dict:
        #         data = {}
        #         data[hash_key] = (random.randint(50, 65), random.randint(50, 65))
        # else:
        data = async_macro_model_train(model_data=darts_neural_dict,
                                       gpus=gpus,
                                       save_dir=save_dir,
                                       dataset=dataset)
        GP_macro_graph_dict[best_dict_keys] = best_graph_list[0]

        # calculate MSE
        pred_v = pred_by_GP(best_graph_list)[0]
        MSE = 0
        for k, v in data.items():
            if k not in GP_macro_graph_dict:
                raise ValueError('model trained acc key should in GP_macro_graph_dict')
            if len(GP_macro_graph_dict[k]) > 2:
                GP_macro_graph_dict[k][2] = v[0]
            else:
                GP_macro_graph_dict[k].extend(v)
            MSE = mean_squared_error([v[0]], [pred_v]) / 10000
        GP_model_keys.append(best_dict_keys)
        fit_GP()
        return copy_population_list, MSE

    # regularized evolution
    num_pop = algo_info['num_pop']
    num_init = algo_info['num_init']
    num_gen = algo_info['num_gen']
    num_resample = algo_info['num_resample']
    mutation_rate = algo_info['mutation_rate']
    crossover_rate = algo_info['crossover_rate']
    encode_path = algo_info['encode_path']
    init_nasbench_macro_cifar10(save_dir)

    def tournament_selection(Pops, Pops_key):
        randint1 = random.randint(0, num_pop - 1)
        randint2 = random.randint(0, num_pop - 1)
        candidate1 = Pops[Pops_key[randint1]]
        candidate2 = Pops[Pops_key[randint2]]
        if candidate1[2] > candidate2[2]:
            return candidate1
        else:
            return candidate2

    def crossover(p1, p2):
        # clean the valid_acc
        p1[2] = 0
        p2[2] = 0
        phenotype1 = p1[0]
        phenotype2 = p2[0]
        crossover_position = random.randint(0, 3)
        cross_phenotype1, cross_phenotype2 = [], []
        for cell in phenotype1:
            cross_phenotype1.append(deepcopy(cell[crossover_position * 2:crossover_position * 2 + 2]))
        for cell in phenotype2:
            cross_phenotype2.append(deepcopy(cell[crossover_position * 2:crossover_position * 2 + 2]))

        # crossover
        for i in range(2):
            phenotype1[i][crossover_position * 2:crossover_position * 2 + 2] = cross_phenotype2[i]
            phenotype2[i][crossover_position * 2:crossover_position * 2 + 2] = cross_phenotype1[i]

        p1[0] = phenotype1
        p2[0] = phenotype2
        return p1, p2

    def op_mutation(p):
        # clean valid_acc
        p[2] = 0

        norm_or_redu = random.randint(0, 1)
        mutation_position = random.randint(0, 7)
        phenotype = deepcopy(p[0])
        # print('change position {}, {}'.format(norm_or_redu, mutation_position))
        connect = phenotype[norm_or_redu][mutation_position][0]
        op = phenotype[norm_or_redu][mutation_position][1]
        new_op = random.choice(list(np.delete(np.arange(8), op)))
        phenotype[norm_or_redu][mutation_position] = (connect, new_op)
        p[0] = phenotype
        return p

    def environment_selection(cur_Pops, cur_evaluated_Offspring, elitism=0.2):
        e_count = int(num_pop * elitism)
        merge_dict = {**cur_Pops, **cur_evaluated_Offspring}
        sort_merge_dict_list = sorted(merge_dict.items(), key=lambda x: x[1][2], reverse=True)
        e_dict = {}
        for i in range(e_count):
            hash_key = sort_merge_dict_list[i][0]
            e_dict[hash_key] = merge_dict[hash_key]

        for _ in range(num_pop - e_count):
            i1 = random.randint(e_count - 1, len(merge_dict) - 1)
            i2 = random.randint(e_count - 1, len(merge_dict) - 1)
            i1_acc = sort_merge_dict_list[i1][1][2]
            i2_acc = sort_merge_dict_list[i2][1][2]
            if i1_acc > i2_acc:
                hash_key = sort_merge_dict_list[i1][0]
            else:
                hash_key = sort_merge_dict_list[i2][0]
            while hash_key in e_dict:
                # random sample
                random_hash_idx = random.randint(0, len(sort_merge_dict_list) - 1)
                hash_key = sort_merge_dict_list[random_hash_idx][0]
            e_dict[hash_key] = merge_dict[hash_key]

        return e_dict

    # set seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.enabled = True
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

    init_pop_list = search_space.generate_random_dataset(num=num_pop, encode_paths=encode_path)
    data_dict_keys = [convert_to_genotype(d[0], verbose=False) for d in init_pop_list]
    data_dict_keys = [sha256(str(k).encode('utf-8')).hexdigest() for k in data_dict_keys]
    cur_Pops = {}
    for i, d in enumerate(init_pop_list):
        cur_Pops[data_dict_keys[i]] = list(d)

    # choose num_init//5 arch to train
    first_init_num = num_init // 5
    first_init_pop_list = deepcopy(init_pop_list[:first_init_num])
    data_dict_keys = [convert_to_genotype(d[0], verbose=False) for d in first_init_pop_list]
    data_dict_keys = [sha256(str(k).encode('utf-8')).hexdigest() for k in data_dict_keys]
    GP_model_keys.extend(data_dict_keys)
    for i, d in enumerate(first_init_pop_list):
        GP_macro_graph_dict[data_dict_keys[i]] = list(d)
    darts_neural_dict = search_space.assemble_cifar10_neural_net(first_init_pop_list)
    data = async_macro_model_train(model_data=darts_neural_dict,
                                   gpus=gpus,
                                   save_dir=save_dir,
                                   dataset=dataset)
    for k, v in data.items():
        if k not in GP_macro_graph_dict:
            raise ValueError('model trained acc key should in GP_macro_graph_dict')
        GP_macro_graph_dict[k].extend(v)

    # train GP and predict the left pops
    # fit GP surrogate
    fit_GP()

    # predict left population, and choose the largest EI to evaluate on GPUS
    remain_pop_list = init_pop_list[first_init_num:]
    while len(GP_macro_graph_dict) < num_init:
        val_acc = np.array([GP_macro_graph_dict[k][2] for k in GP_model_keys])
        remain_pop_list, _ = updata_GP(remain_pop_list, val_acc)

    # predict the left population by GP, and give them pred_ACC
    pred_results = pred_by_GP(remain_pop_list)

    for hash_key in GP_macro_graph_dict:
        cur_Pops[hash_key].append(GP_macro_graph_dict[hash_key][2])

    data_dict_keys = [convert_to_genotype(d[0], verbose=False) for d in remain_pop_list]
    data_dict_keys = [sha256(str(k).encode('utf-8')).hexdigest() for k in data_dict_keys]
    for i, hash_key in enumerate(data_dict_keys):
        cur_Pops[hash_key].append(pred_results[i])

    cur_Pops_key_list = []
    for hash_key in cur_Pops:
        cur_Pops[hash_key][1] = None
        cur_Pops_key_list.append(hash_key)

    gen_no = 0
    while gen_no < num_gen:
        gen_no += 1
        cur_Offspring = []
        MSE_list_ = []
        # start to generate offspring
        for _ in range(num_pop // 2):
            p1 = deepcopy(tournament_selection(cur_Pops, cur_Pops_key_list))
            p2 = deepcopy(tournament_selection(cur_Pops, cur_Pops_key_list))
            # start crossover
            if random.random() < crossover_rate:
                # crossover
                p1, p2 = crossover(p1, p2)

            # start mutation
            # start op_type mutation
            if random.random() < mutation_rate:
                p1 = op_mutation(p1)
            if random.random() < mutation_rate:
                p2 = op_mutation(p2)

            # add into offspring
            cur_Offspring.append(p1)
            cur_Offspring.append(p2)

        data_dict_keys = [convert_to_genotype(d[0], verbose=False) for d in cur_Offspring]
        data_dict_keys = [sha256(str(k).encode('utf-8')).hexdigest() for k in data_dict_keys]
        cur_evaluated_Offspring = {}
        for i, d in enumerate(cur_Offspring):
            cur_evaluated_Offspring[data_dict_keys[i]] = list(d)

        # start to predict acc and update GP
        for _ in range(num_resample):
            val_acc = np.array([GP_macro_graph_dict[k][2] for k in GP_model_keys])
            cur_Offspring, MSE = updata_GP(cur_Offspring, val_acc)
            MSE_list_.append(MSE)

        # predict remain offspring by GP
        pred_results = pred_by_GP(cur_Offspring)

        data_dict_keys = [convert_to_genotype(d[0], verbose=False) for d in cur_Offspring]
        data_dict_keys = [sha256(str(k).encode('utf-8')).hexdigest() for k in data_dict_keys]
        for i, hash_key in enumerate(data_dict_keys):
            cur_evaluated_Offspring[hash_key][2] = pred_results[i]

        for hash_key in GP_macro_graph_dict:
            if hash_key in cur_evaluated_Offspring:
                cur_evaluated_Offspring[hash_key][2] = GP_macro_graph_dict[hash_key][2]

        cur_Pops = environment_selection(cur_Pops, cur_evaluated_Offspring)

        cur_Pops_key_list = []
        for hash_key in cur_Pops:
            cur_Pops[hash_key][1] = None
            cur_Pops_key_list.append(hash_key)

        if verbose:
            logger.info('Gen: {}, mean MSE for resampled architectures: {}'.format(gen_no, np.mean(MSE_list_)))
            logger.info('Number of GP_Archive: {}'.format(len(GP_macro_graph_dict)))

        pop_save_path = save_dir + '/population_pkl/' + str(gen_no) + '.pkl'
        with open(pop_save_path, 'wb') as file:
            pickle.dump(cur_Pops, file)
            pickle.dump(cur_Pops_key_list, file)
    return GP_macro_graph_dict, GP_model_keys
