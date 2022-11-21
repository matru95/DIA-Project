import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from TS_Learner import *


class SWTS_Learner(TS_Learner):
    def __init__(self, subcamp_id, n_arms, arms, window_size):
        super(SWTS_Learner, self).__init__(subcamp_id, n_arms, arms)
        self.window_size = window_size
        self.rewards_per_arm_full = [[] for i in range(n_arms)]

    def update_observations_sw(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        for a in range(self.n_arms):
            if a == pulled_arm:
                self.rewards_per_arm_full[pulled_arm].append(reward)
            else:
                self.rewards_per_arm_full[a].append(np.nan)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations_sw(pulled_arm, reward)
        cum_rew = np.nansum(self.rewards_per_arm_full[pulled_arm][-self.window_size:]) 
        #n_rounds_arm = len(self.rewards_per_arm_full[pulled_arm][-self.window_size:])
        n_rounds_arm = 0
        for x in self.rewards_per_arm_full[pulled_arm][-self.window_size:]:
            if not np.isnan(x):
                n_rounds_arm += 1
        self.beta_parameters[pulled_arm, 0] = cum_rew + 1.0
        self.beta_parameters[pulled_arm, 1] = n_rounds_arm - cum_rew + 1.0