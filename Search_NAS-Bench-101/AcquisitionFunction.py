import numpy as np
from scipy.stats import norm


# this is copy from https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py
def expected_improvement(x, gaussian_process, evaluated_y, greater_is_better=True, xi=0, n_params=56):
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


# maximizing the predicted Objective Function
def maximizing_predicted(x, gaussian_process, n_params=56):
    x_to_predict = np.array(x).reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    return mu


def probability_improvement(x, gaussian_process, evaluated_y, xi=0, n_params=56):
    x_to_predict = np.array(x).reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    optimum_value = np.max(evaluated_y)

    Z = (mu - optimum_value - xi) / sigma
    probability_improvement = norm.cdf(Z)

    return -1 * probability_improvement


def upper_confidence_bound(x, gaussian_process, kappa=1, n_params=56):
    x_to_predict = np.array(x).reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    upper_confidence_bound = mu + sigma * kappa
    return -1 * upper_confidence_bound

def lower_confidence_bound(x, gaussian_process, kappa=1, n_params=56):
    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    lower_confidence_bound = mu - sigma * kappa
    return lower_confidence_bound