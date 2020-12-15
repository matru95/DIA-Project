import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
np.random.seed(22)

def build_real_curve(x, param):
    coeff = param[1] / param[0]
    return np.maximum(np.minimum(x*coeff, param[1]) - 1, 0)

def generate_observations(x, noise_std, param):
    return build_real_curve(x, param) + np.random.normal(0, noise_std, size = build_real_curve(x, param).shape)

class BudgetEstimator:

    def __init__(self, n_obs, exp_budgets, noise_std, param):
        x_obs = np.array([])
        y_obs = np.array([])

        for i in range(0, n_obs):
            new_x_obs = np.random.choice(exp_budgets,1)
            new_y_obs = generate_observations(new_x_obs, noise_std, param)

            x_obs = np.append(x_obs, new_x_obs)
            y_obs = np.append(y_obs, new_y_obs)
            
        X = np.atleast_2d(x_obs).T
        Y = y_obs.ravel()

        theta = 1.0
        l = 1.0
        kernel = C(theta, (1e-3,1e3)) * RBF(l, (1e-3,1e3))
        self.gp = GaussianProcessRegressor(kernel = kernel, 
                                        alpha = noise_std**2, 
                                        normalize_y = True,
                                        n_restarts_optimizer = 10)
        self.gp.fit(X, Y)
        self.x_pred = np.atleast_2d(exp_budgets).T
        self.y_pred, self.sigma = self.gp.predict(self.x_pred, return_std = True)
        '''
        plt.figure(0)
        plt.plot(self.x_pred, build_real_curve(self.x_pred, param), 'r:', label = 'True Curve')
        plt.plot(X.ravel(), Y, 'ro', label = u'Observed Clicks')
        plt.plot(self.x_pred, self.y_pred, 'b-', label = u'Predicted Clicks')
        plt.fill(np.concatenate([self.x_pred, self.x_pred[::-1]]),
                    np.concatenate([self.y_pred - 1.96 * self.sigma, (self.y_pred + 1.96 * self.sigma)[::-1]]),
                    alpha = 0.5, fc = 'b', ec = 'None', label = '95%\\ conf interval')
        plt.xlabel('budget')
        plt.ylabel("Number of Clicks")
        plt.legend(loc = 'lower right')
        plt.savefig('pred_Business.png')
        plt.show()

n_obs = 50
budgets = np.linspace(0.0, 50.0, n_obs)
noise_std = 2.0
param = (8, 17)
best = BudgetEstimator(n_obs, budgets, noise_std, param)
#best.gp.predict(x)
'''