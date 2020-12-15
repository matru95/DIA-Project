import numpy as np
from BudgetEstimator import *
import matplotlib.pyplot as plt

def plot_curve(X, Y_pred, Y_real):
    plt.figure(1)
    plt.plot(X,Y_pred, c='g')
    plt.plot(X,Y_real, c='r', ls=":")
    plt.show()

def estimate_curve(budgets, param):
    estimation_noise_std = 0.5
    buds = np.linspace(0.0, 50.0, 50)
    best = BudgetEstimator(len(buds), buds, estimation_noise_std, param)
    plot_curve(buds, np.maximum(0,best.gp.predict(np.atleast_2d(buds).T)), build_real_curve(buds, param))
    return np.maximum(0,best.gp.predict(np.atleast_2d(budgets).T))

class BudgetEnvironment:

    def __init__(self, subcamp_id, budgets, sigma, params):
        self.subcamp_id = subcamp_id
        self.budgets = budgets
        self.phases = []
        self.sigmas = np.ones(len(budgets)) * sigma
        for p in params:
            means = estimate_curve(budgets, p)
            self.phases.append(means)

    def round(self, pulled_arm):
        return np.random.normal(self.phases[0][pulled_arm], 
                                self.sigmas[pulled_arm])


