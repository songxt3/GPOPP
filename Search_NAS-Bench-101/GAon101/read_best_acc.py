import pickle

if __name__ == '__main__':
    m_prob = 0.2
    population_size = 100
    num_generation = 15
    ac_function = 'UCB'
    num_resample = 5
    first_archive_size = 50
    save_log = False
    save_acc = True
    repeat_times = 100

    save_path = r'../pkl/best_acc/best_acc_pop{}_gen{}_Mpro{}_repeat{}_numResample{}_numInit{}_ac-{}.pkl'.format(
            population_size, num_generation, m_prob, repeat_times, num_resample, first_archive_size, ac_function)
    with open(save_path, 'rb') as file:
        load_dic = pickle.load(file)