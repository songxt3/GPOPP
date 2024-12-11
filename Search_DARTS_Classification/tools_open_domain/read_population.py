import pickle

pkl_path = '../train_output/open_domain_darts_gp_1/population_pkl/'

gen_no = 15

with open(pkl_path + str(gen_no) + '.pkl', 'rb') as file:
    cur_Pops = pickle.load(file)
    cur_Pops_key_list = pickle.load(file)
    print()