import numpy as np
from Learner import *


class TS_Learner(Learner):
    def __init__(self, subcamp_id, n_arms, arms):
        super(TS_Learner, self).__init__(n_arms)
        self.subcamp_id = subcamp_id
        self.arms = arms
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] += reward
        self.beta_parameters[pulled_arm, 1] += (1.0-reward)
    
    def draw_samples(self):
        return np.random.beta([item[0] for item in self.beta_parameters], [item[1] for item in self.beta_parameters]) 
    
    def beta_mean(self, pulled_arm):
        return self.beta_parameters[pulled_arm][0] / (self.beta_parameters[pulled_arm][0] + self.beta_parameters[pulled_arm][1])