import numpy as np
import matplotlib.pyplot as plt

from shared.PriceEnvironment import *
from shared.TS_Learner import *


x = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70]
y1 = [0.4, 0.45, 0.55, 0.65, 0.5, 0.35, 0.25, 0.1, 0.08, 0.05, 0.02, 0.001]
y2 = [0.3, 0.45, 0.55, 0.65, 0.69, 0.72, 0.59, 0.35, 0.27, 0.13, 0.04, 0.001]
y3 = [0.1, 0.17, 0.24, 0.33, 0.46, 0.57, 0.63, 0.58, 0.3, 0.1, 0.02, 0.001]

p = [0.1, 0.8, 0.1]
y_aggr = list(map(lambda x, y, z: (x*p[0]+y*p[1]+z*p[2]), y1, y2, y3))

n_arms = 6
prices = np.linspace(min(x), max(x), num=n_arms, endpoint=True)
num_of_subcamp = 3
T = 366
n_experiments = 1000
rewards_per_experiment = [[] for _ in range(n_experiments)]

enviroments = np.array([PriceEnvironment(subcamp_id = None, n_arms=n_arms, arms=prices, x=x, y=y_aggr)])

opt = enviroments[0].optimum

for e in range(n_experiments):
    
    learner = TS_Learner(subcamp_id = None, n_arms=n_arms, arms=prices)

    for t in range(T):         
        pulled_arm = learner.pull_arm()
        reward = enviroments[0].round(pulled_arm)
        learner.update(pulled_arm, reward)
    rewards_per_experiment[e].append(learner.collected_rewards)

rewards = np.mean(rewards_per_experiment, axis=0)[0]
plt.figure(0)
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(rewards, 'r')
plt.plot(np.full(T, opt), '--k')
plt.legend(["TS", "Optimum"])
plt.show()

regrets = np.cumsum(np.mean(np.full(T, opt) - rewards_per_experiment, axis=0))
plt.figure(1)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(regrets, 'b')
plt.show()