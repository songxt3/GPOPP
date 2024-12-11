import sys
import os
sys.path.append(os.getcwd())
from GAon101.GP_Surrogate_evolve import genotype2phenotype, get_input_X
from GAon101.evolve import Evolution, query_fitness, query_index
from GAon101.population import Population
from e2epp.e2epp import train_e2epp, test_e2epp
from GCNpredictor.predict import train_gcn, test_gcn
from get_data_from_101 import delete_margin
from GAon101.utils import operations2onehot
import numpy as np
import pickle
import argparse


class Offline_Evolution(Evolution):
    def __init__(self, surrogate_name, m_prob=0.2, m_num_matrix=1, m_num_op_list=1, x_prob=0.9, population_size=100):
        super(Offline_Evolution, self).__init__(m_prob, m_num_matrix, m_num_op_list, x_prob, population_size)
        surrogate_name_list = ['e2epp', 'GCN']
        if surrogate_name not in surrogate_name_list:
            raise ValueError('The surrogate name must be in {}'.format(str(surrogate_name_list)))
        self.surrogate_name = surrogate_name

    def init_train_surrogate(self, pops: Population):
        if self.surrogate_name == 'e2epp':
            query_fitness(0, pops)
            geno_pops = genotype2phenotype(pops)
            X = []
            y = []
            for indi in geno_pops.pops:
                input_metrix = []
                matrix = indi.indi['matrix']
                matrix = delete_margin(matrix)
                flattend_matrix = np.reshape(matrix, (-1)).tolist()
                input_metrix.extend(flattend_matrix)

                op_list = indi.indi['op_list']
                op_list = operations2onehot(op_list[1: -1])
                input_metrix.extend(op_list)

                X.append(input_metrix)
                y.append(indi.mean_acc)
            tree, features = train_e2epp(X, y)
            self.surrogate = {'tree': tree, 'features': features}
        else:  # GCN
            train_index = query_index(pops)
            self.surrogate = train_gcn(train_index)

    def predict_by_surrogate(self, pred_pops: Population):
        if self.surrogate_name == 'e2epp':
            pred_pops_new = genotype2phenotype(pred_pops)
            X = get_input_X(pred_pops_new)
            tree = self.surrogate['tree']
            features = self.surrogate['features']
            pred_y = test_e2epp(X, tree, features)
            for i, indi in enumerate(pred_pops.pops):
                pred_pops.pops[i].mean_acc = pred_y[i]
                # using the GP model, don't need to training
                pred_pops.pops[i].mean_training_time = 0
        else:  # GCN
            test_index = query_index(pred_pops)
            pred_y = test_gcn(test_index, self.surrogate)
            for i, indi in enumerate(pred_pops.pops):
                pred_pops.pops[i].mean_acc = pred_y[i]
                # using the GP model, don't need to training
                pred_pops.pops[i].mean_training_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Offline predictor')
    parser.add_argument('--predictor', type=str, choices=['GCN', 'e2epp'], default='e2epp', help='name of predictor')
    args = parser.parse_args()

    m_prob = 0.2
    m_num_matrix = 1
    m_num_op_list = 1
    x_prob = 0.9
    population_size = 100
    total_train_size = 125
    num_generation = 15
    save_log = False
    save_acc = True
    repeat_times = 100
    # 'e2epp' or 'GCN'
    surrogate_name = args.predictor

    final_acc_list_ = []
    final_wall_time_list_ = []
    Offline_Evolution = Offline_Evolution(surrogate_name, m_prob, m_num_matrix, m_num_op_list, x_prob, population_size)
    for _ in range(repeat_times):
        Offline_Evolution.initialize_popualtion()
        gen_no = 0
        query_fitness(gen_no, Offline_Evolution.pops, save_log=save_log)
        total_train_population = Population(total_train_size - population_size)
        query_fitness(-2, total_train_population, save_log=save_log)
        total_train_population.merge_populations(Offline_Evolution.pops.pops)
        total_training_time = total_train_population.calculate_population_training_time()
        Offline_Evolution.init_train_surrogate(total_train_population)

        while True:
            gen_no += 1
            if gen_no > num_generation:
                break
            offsprings = Offline_Evolution.recombinate(population_size)
            Offline_Evolution.predict_by_surrogate(offsprings)
            Offline_Evolution.environmental_selection(gen_no, offsprings, save_log)

        # for the last generation
        # last_resample_num = 1
        sorted_acc_index = Offline_Evolution.pops.get_sorted_index_order_by_acc()
        last_population = Population(0)
        # for i in sorted_acc_index[:last_resample_num]:
        best_individual = Offline_Evolution.pops.get_individual_at(0)
        last_population.add_individual(best_individual)

        gen_no = 'final'
        query_fitness(gen_no, last_population, 'final_test_accuracy', save_log)

        final_acc_list_.append(best_individual.mean_acc)
        final_wall_time_list_.append(total_training_time)

    print('Population size: {}, gen: {}, repeat: {}, m_pro: {}, train_size: {}, offline-{}'.format(
        population_size, num_generation, repeat_times, m_prob, total_train_size, surrogate_name))
    print('Final ACC: Mean:{}, Std:{}'.format(np.mean(final_acc_list_), np.std(final_acc_list_)))
    print('Final wall time: Mean:{}, Std:{}'.format(np.mean(final_wall_time_list_), np.std(final_wall_time_list_)))

    if save_acc:
        acc_save_path = r'../pkl/best_acc/best_acc_pop{}_gen{}_Mpro{}_repeat{}_total_query{}_offline-{}.pkl'. \
            format(population_size, num_generation, m_prob, repeat_times, total_train_size, surrogate_name)
        query_number = total_train_size
        save_dic = {'final_acc_list_': final_acc_list_, 'final_wall_time_list_': final_wall_time_list_,
                    'query_number': query_number}
        with open(acc_save_path, 'wb') as file:
            pickle.dump(save_dic, file)
