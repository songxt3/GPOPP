import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF, RationalQuadratic, ExpSineSquared, DotProduct
from Toy_experiment import get_toy_metrics, get_toy_data, get_elitism_metrics, get_sorted_training_data, try_GP_method
from AcquisitionFunction import expected_improvement, probability_improvement

kernel = Matern()
GPmodel = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
method_name = 'GP'

if __name__ == '__main__':
    train_num = 1000
    test_num = 5000
    additional_metrics = False
    integers2one_hot = True
    more_train_data = False
    elitism = True
    num_resample = 50

    metrics = get_toy_metrics(train_num)
    print('----------------------train---------------------')

    X, y, _ = get_toy_data(metrics, create_more_metrics=more_train_data, select_upper_tri=False,
                           additional_metrics=additional_metrics, integers2one_hot=integers2one_hot)

    print('----------------------test----------------------')
    if elitism:
        test_metrics = get_elitism_metrics(elitism=0.8, dataset_size=10000)
    else:
        test_metrics = get_toy_metrics(test_num, type='fixed_test', train_num=train_num)
    testX, testy, num_new_metrics = get_toy_data(test_metrics, create_more_metrics=False,
                                                 select_upper_tri=False,
                                                 additional_metrics=additional_metrics,
                                                 integers2one_hot=integers2one_hot)

    new_X, new_y = X[:100], y[:100]
    for i in range(9):
        index_start = (i + 1) * 100
        next_pop_X = np.array(X[index_start:index_start + 100])
        next_pop_y = np.array(y[index_start:index_start + 100])
        for j in range(num_resample):
            GPmodel.fit(new_X, new_y)
            PI = probability_improvement(next_pop_X[j:100], GPmodel, new_y, xi=0.25)
            # PI is negative
            sorted_PI_index = np.argsort(PI)
            best_PI_index = sorted_PI_index[0]
            next_pop_X[[j, best_PI_index+j]] = next_pop_X[[best_PI_index+j, j]]
            next_pop_y[[j, best_PI_index+j]] = next_pop_y[[best_PI_index+j, j]]
            new_X = new_X + list([list(next_pop_X[j])])
            new_y = new_y + list([next_pop_y[j]])
        print(len(new_y))
        _, _ = try_GP_method(new_X, new_y, testX, testy, GPmodel, method_name, show_fig=False)
