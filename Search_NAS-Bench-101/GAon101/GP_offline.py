from GAon101.GP_Surrogate_evolve import *

if __name__ == '__main__':
    m_prob = 0.2
    m_num_matrix = 1
    m_num_op_list = 1
    x_prob = 0.9
    population_size = 100
    total_train_size = 175
    num_generation = 15
    ac_function = 'EI'
    kernel = Matern()
    num_resample = 0
    save_log = False
    save_acc = True
    repeat_times = 100

    final_acc_list_ = []
    final_wall_time_list_ = []
    GP_Evolution = GP_evolution(ac_function, kernel, m_prob, m_num_matrix, m_num_op_list, x_prob, population_size)
    warnings.filterwarnings("ignore")
    for _ in range(repeat_times):
        GP_Evolution.initialize_popualtion()
        GP_Evolution.init_GP_model()
        gen_no = 0
        query_fitness(gen_no, GP_Evolution.pops, save_log=save_log)
        total_train_population = Population(total_train_size - population_size)
        query_fitness(-2, total_train_population, save_log=save_log)
        total_train_population.merge_populations(GP_Evolution.pops.pops)
        total_training_time = total_train_population.calculate_population_training_time()

        GP_Evolution.GP_archive = genotype2phenotype(total_train_population)
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

        # for the last generation
        # last_resample_num = 1
        sorted_acc_index = GP_Evolution.pops.get_sorted_index_order_by_acc()
        last_population = Population(0)
        # for i in sorted_acc_index[:last_resample_num]:
        best_individual = GP_Evolution.pops.get_individual_at(0)
        last_population.add_individual(best_individual)

        gen_no = 'final'
        query_fitness(gen_no, last_population, 'final_test_accuracy', save_log)


        if save_log:
            population_log(gen_no, last_population)

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
        final_wall_time_list_.append(total_training_time)

    query_num = len(GP_Evolution.GP_archive.pops)
    print('Population size: {}, gen: {}, repeat: {}, m_pro: {}, AC: {}, resample: {}'.format(
        population_size, num_generation, repeat_times, m_prob, ac_function, num_resample))
    print('Final ACC: Mean:{}, Std:{}'.format(np.mean(final_acc_list_), np.std(final_acc_list_)))
    print('Final wall time: Mean:{}, Std:{}'.format(np.mean(final_wall_time_list_), np.std(final_wall_time_list_)))

    if save_acc:
        acc_save_path = r'../pkl/best_acc/best_acc_pop{}_gen{}_Mpro{}_repeat{}_numResample{}.pkl'.format(
            population_size, num_generation, m_prob, repeat_times, num_resample)
        query_number = population_size + num_resample * num_generation
        save_dic = {'final_acc_list_': final_acc_list_, 'final_wall_time_list_': final_wall_time_list_,
                    'query_number': query_number}
        with open(acc_save_path, 'wb') as file:
            pickle.dump(save_dic, file)