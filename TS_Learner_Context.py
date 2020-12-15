import numpy as np
from TS_Learner import *

class TS_Learner_Context(TS_Learner):
    def __init__(self, subcamp_id, n_arms, arms):
        super(TS_Learner_Context, self).__init__(subcamp_id, n_arms, arms)
        self.number_of_pulls = np.zeros(n_arms)

    def is_pulled(self, pulled_arm):
        a = self.number_of_pulls[pulled_arm]
        if int(a) == 0: return False 
        else: return True

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] += reward
        self.beta_parameters[pulled_arm, 1] += (1.0-reward)
        self.number_of_pulls[pulled_arm] += 1