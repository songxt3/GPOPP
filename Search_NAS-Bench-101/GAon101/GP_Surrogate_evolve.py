from GAon101.evolve import Evolution, query_fitness, query_fitness_for_indi
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from GAon101.population import Population
import numpy as np
import copy
from get_data_from_101 import delete_margin
from GAon101.utils import operations2onehot, GP_log, population_log, NULL
from AcquisitionFunction import expected_improvement, probability_improvement, upper_confidence_bound, \
    maximizing_predicted
from nasbench import api
from GAon101.individual import Individual
import pickle
import random
import warnings
from scipy.stats import kendalltau


# transform genotype population to phenotype, and pad the matrix and the op_list to max length
def genotype2phenotype(pops: Population) -> Population:
    genotype_population = pops
    phenotype_population = Population(0)
    for indi in genotype_population.pops:
        matrix = indi.indi['matrix']
        op_list = indi.indi['op_list']
        model_spec = api.ModelSpec(matrix, op_list)
        pruned_matrix = model_spec.matrix
        pruned_op_list = model_spec.ops
        # start to padding zeros to 7*7 matrix and 7 op_list
        len_operations = len(pruned_op_list)
        assert len_operations == len(pruned_matrix)
        padding_matrix = copy.deepcopy(pruned_matrix)
        if len_operations != 7:
            for j in range(len_operations, 7):
                pruned_op_list.insert(j - 1, NULL)

            padding_matrix = np.insert(pruned_matrix, len_operations - 1,
                                       np.zeros([7 - len_operations, len_operations]), axis=0)
            padding_matrix = np.insert(padding_matrix, [len_operations - 1], np.zeros([7, 7 - len_operations]), axis=1)
        phenotype_individual = Individual()
        phenotype_individual.create_an_individual(padding_matrix, pruned_op_list)
        phenotype_individual.mean_acc = indi.mean_acc
        phenotype_individual.mean_training_time = indi.mean_training_time
        phenotype_population.add_individual(phenotype_individual)
    return phenotype_population


def get_input_X(pops: Population):
    X = []
    for indi in pops.pops:
        input_metrix = []
        matrix = indi.indi['matrix']
        matrix = delete_margin(np.array(matrix))
        flattend_matrix = np.reshape(matrix, (-1)).tolist()
        input_metrix.extend(flattend_matrix)

        op_list = indi.indi['op_list']
        op_list = operations2onehot(op_list[1: -1])
        input_metrix.extend(op_list)

        X.append(input_metrix)
    return X


# this is for genotype surrogate evolution
class GP_evolution(Evolution):
    def __init__(self, ac_function, kernel=Matern(), m_prob=0.2, m_num_matrix=1, m_num_op_list=1, x_prob=0.9,
                 population_size=100):
        super(GP_evolution, self).__init__(m_prob, m_num_matrix, m_num_op_list, x_prob, population_size)
        self.ac_function = ac_function
        self.kernel = kernel
        self.GP_archive = Population(0)

    def init_GP_model(self):
        self.GPmodel = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True)

    def update_GP_archive(self, new_indi):
        self.GP_archive.add_individual(new_indi)

    def fit_GP_model(self):
        X = []
        y = []
        for indi in self.GP_archive.pops:
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
        self.GPmodel.fit(X, y)

    def predict_by_GP(self, pred_pops: Population):
        pred_pops_new = genotype2phenotype(pred_pops)
        X = get_input_X(pred_pops_new)
        pred_y = self.GPmodel.predict(X)
        for i, indi in enumerate(pred_pops.pops):
            pred_pops.pops[i].mean_acc = pred_y[i]
            # using the GP model, don't need to training
            pred_pops.pops[i].mean_training_time = 0


if __name__ == '__main__':
    m_prob = 0.2
    m_num_matrix = 1
    m_num_op_list = 1
    x_prob = 0.9
    population_size = 100
    num_generation = 15
    ac_function = 'MP+EI'
    kernel = Matern()
    num_resample = 5
    first_archive_size = 50
    save_log = True
    save_acc = True
    repeat_times = 100

    final_acc_list_ = []
    ktau_list_ = []
    final_wall_time_list_ = []
    GP_Evolution = GP_evolution(ac_function, kernel, m_prob, m_num_matrix, m_num_op_list, x_prob, population_size)
    warnings.filterwarnings("ignore")
    for _ in range(repeat_times):
        GP_Evolution.initialize_popualtion()
        GP_Evolution.init_GP_model()
        gen_no = 0
        random.shuffle(GP_Evolution.pops.pops)
        first_random_sample_size = first_archive_size // 5
        initial_archive_pop = Population(0)
        initial_archive_pop.set_populations(GP_Evolution.pops.pops[:first_random_sample_size])
        query_fitness(gen_no, initial_archive_pop)
        GP_Evolution.GP_archive = genotype2phenotype(initial_archive_pop)
        GP_Evolution.fit_GP_model()
        # choosing the remaining initial individuals to evaluate
        remain_pop = Population(0)
        remain_pop.set_populations(GP_Evolution.pops.pops[first_random_sample_size:])
        pred_init_individuals = genotype2phenotype(remain_pop)
        for re_i in range(first_random_sample_size, first_archive_size):
            # select the ac function, input pred_offspring
            if GP_Evolution.ac_function == 'PI':
                criterion_metric = probability_improvement(get_input_X(pred_init_individuals), GP_Evolution.GPmodel,
                                                           GP_Evolution.GP_archive.get_best_acc())
            elif GP_Evolution.ac_function == 'UCB':
                criterion_metric = upper_confidence_bound(get_input_X(pred_init_individuals), GP_Evolution.GPmodel)
            elif GP_Evolution.ac_function == 'MP': # for maximizing predicted objective function
                criterion_metric = maximizing_predicted(get_input_X(pred_init_individuals), GP_Evolution.GPmodel)
            elif GP_Evolution.ac_function == 'MP+EI':
                # random choosing MP or EI for each sample
                choice = random.choice(('MP', 'EI'))
                if choice == 'MP':
                    criterion_metric = maximizing_predicted(get_input_X(pred_init_individuals), GP_Evolution.GPmodel)
                elif choice == 'EI':
                    criterion_metric = expected_improvement(get_input_X(pred_init_individuals), GP_Evolution.GPmodel,
                                                            GP_Evolution.GP_archive.get_best_acc())
                else: raise IndexError
            else:  # GP_Evolution.ac_function == 'EI' or others
                criterion_metric = expected_improvement(get_input_X(pred_init_individuals), GP_Evolution.GPmodel,
                                                        GP_Evolution.GP_archive.get_best_acc())
            # the criterion is negative
            sorted_index = np.argsort(criterion_metric)
            best_index = sorted_index[0]
            pop_pred_indi = pred_init_individuals.pops.pop(best_index)
            query_fitness_for_indi(pop_pred_indi)
            # update GP_archive
            query_fitness_for_indi(pop_pred_indi)
            GP_Evolution.update_GP_archive(pop_pred_indi)
            GP_Evolution.fit_GP_model()
        while True:
            gen_no += 1
            if gen_no > num_generation:
                break
            offspring = GP_Evolution.recombinate(population_size)
            pred_offspring = genotype2phenotype(offspring)
            query_offspring = Population(0)
            # update GP model
            for re_i in range(num_resample):
                # select the ac function, input pred_offspring
                if GP_Evolution.ac_function == 'PI':
                    criterion_metric = probability_improvement(get_input_X(pred_offspring), GP_Evolution.GPmodel,
                                                               GP_Evolution.GP_archive.get_best_acc())
                elif GP_Evolution.ac_function == 'UCB':
                    criterion_metric = upper_confidence_bound(get_input_X(pred_offspring), GP_Evolution.GPmodel)
                elif GP_Evolution.ac_function == 'MP':  # for maximizing predicted objective function
                    criterion_metric = maximizing_predicted(get_input_X(pred_init_individuals), GP_Evolution.GPmodel)
                elif GP_Evolution.ac_function == 'MP+EI':
                    # random choosing MP or EI for each sample
                    choice = random.choice(('MP', 'EI'))
                    if choice == 'MP':
                        criterion_metric = maximizing_predicted(get_input_X(pred_init_individuals),
                                                                GP_Evolution.GPmodel)
                    elif choice == 'EI':
                        criterion_metric = expected_improvement(get_input_X(pred_init_individuals),
                                                                GP_Evolution.GPmodel,
                                                                GP_Evolution.GP_archive.get_best_acc())
                    else:
                        raise IndexError
                else:  # GP_Evolution.ac_function == 'EI' or others
                    criterion_metric = expected_improvement(get_input_X(pred_offspring), GP_Evolution.GPmodel,
                                                            GP_Evolution.GP_archive.get_best_acc())
                # the criterion is negative
                sorted_index = np.argsort(criterion_metric)
                best_index = sorted_index[0]
                pop_indi = offspring.pops.pop(best_index)
                pop_pred_indi = pred_offspring.pops.pop(best_index)
                query_fitness_for_indi(pop_indi)
                query_offspring.add_individual(pop_indi)
                # update GP_archive
                query_fitness_for_indi(pop_pred_indi)
                GP_Evolution.update_GP_archive(pop_pred_indi)
                GP_Evolution.fit_GP_model()
            GP_Evolution.predict_by_GP(offspring)
            if save_log:
                GP_log(gen_no, query_offspring, offspring)
            offspring.merge_populations(query_offspring.pops)
            GP_Evolution.environmental_selection(gen_no, offspring, save_log)

        # calculate ktau
        gen_no = 'final'
        last_population_pred_acc = [indi.mean_acc for indi in GP_Evolution.pops.pops]
        last_population_qry = Population(0)
        last_population_qry.set_populations(GP_Evolution.pops.pops)
        query_fitness(gen_no, last_population_qry, 'final_test_accuracy', save_log)
        last_population_qry_acc = [indi.mean_acc for indi in last_population_qry.pops]
        ktau = kendalltau(last_population_pred_acc, last_population_qry_acc)

        # # for the last generation
        # # last_resample_num = 1
        # sorted_acc_index = GP_Evolution.pops.get_sorted_index_order_by_acc()
        # last_population = Population(0)
        # # for i in sorted_acc_index[:last_resample_num]:
        # # because 0 is the best, and is added into the population first
        best_individual = GP_Evolution.pops.get_individual_at(0)
        # last_population.add_individual(best_individual)
        #
        # gen_no = 'final'
        # query_fitness(gen_no, last_population, 'final_test_accuracy', save_log)

        total_training_time = 0
        for indi in GP_Evolution.GP_archive.pops:
            total_training_time += indi.mean_training_time

        if save_log:
            population_log(gen_no, last_population_qry)

            save_path = r'pops_log\total_training_time.txt'
            with open(save_path, 'w') as file:
                file.write('Total_training_time: ' + str(total_training_time) + '\n')
                file.write('Total_training_num: ' + str(population_size + num_generation * num_resample) + '\n')
                file.write(
                    'm_prob: {}, m_num_matrix: {}, m_num_op_list: {}, x_prob: {}\n'.format(m_prob, m_num_matrix,
                                                                                           m_num_op_list,
                                                                                           x_prob))
                file.write('GP_surrogate: True, num_resample: {}, phenotype: {}\n'.format(num_resample, True))
                file.write('AC_function: {}'.format(ac_function))

        final_acc_list_.append(best_individual.mean_acc)
        # ktau[0] maybe nan
        if not np.isnan(ktau[0]):
            ktau_list_.append(ktau[0])
        final_wall_time_list_.append(total_training_time)


    query_num = len(GP_Evolution.GP_archive.pops)
    print('Population size: {}, gen: {}, repeat: {}, m_pro: {}, AC: {}, resample: {}, init: {}, query_num: {}'.format(
        population_size, num_generation, repeat_times, m_prob, ac_function, num_resample, first_archive_size,
        query_num))
    print('Final ACC: Mean:{}, Std:{}'.format(np.mean(final_acc_list_), np.std(final_acc_list_)))
    # print(ktau_list_)
    print('KTau: Mean:{}, Std:{}'.format(np.mean(ktau_list_), np.std(ktau_list_)))
    print('Final wall time: Mean:{}, Std:{}'.format(np.mean(final_wall_time_list_), np.std(final_wall_time_list_)))

    if save_acc:
        acc_save_path = r'../pkl/best_acc/best_acc_pop{}_gen{}_Mpro{}_repeat{}_numResample{}_numInit{}_ac-{}.pkl'.format(
            population_size, num_generation, m_prob, repeat_times, num_resample, first_archive_size, ac_function)
        query_number = population_size + num_resample * num_generation
        save_dic = {'final_acc_list_': final_acc_list_, 'final_wall_time_list_': final_wall_time_list_,
                    'query_number': query_number}
        with open(acc_save_path, 'wb') as file:
            pickle.dump(save_dic, file)
