import GAon101.evolve as evo

if __name__ == '__main__':
    m_prob = 0.2
    m_num_matrix = 1
    m_num_op_list = 1
    x_prob = 0.9
    population_size = 100
    num_generation = 15
    save_log = False
    save_acc = True
    repeat_times = 100

    Evolution = evo.Evolution(m_prob, m_num_matrix, m_num_op_list, x_prob, population_size)
    Evolution.initialize_popualtion(random_initial=True, save_log=save_log)
    query_index = evo.query_index(Evolution.pops)
    print()