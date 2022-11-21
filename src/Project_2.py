import numpy as np
import matplotlib.pyplot as plt

from shared.BudgetEnvironment import *
from shared.GPTS_Learner import *
from shared.Knapsack import *

def generate_table(learners):
    learners = np.atleast_1d(learners)
    table = np.array(list(map(lambda x: x.draw_samples(), learners)))
    table[:, 0] = 0
    return table

def curve_table(envs):
    envs = np.atleast_1d(envs)
    return np.array(list(map(lambda x: x.phases[0], envs)))

n_arms = 6
min_budget = 0.0
max_budget = 50.0
budgets = np.linspace(min_budget, max_budget, n_arms)
sigma = 2

daily_budget = 50.0
num_of_subcamp = 3

T = int(366 / 3)
n_experiments = 10
rewards_per_experiment = [np.zeros(T) for _ in range(n_experiments)]
normalized_rewards_per_experiment = [np.zeros(T) for _ in range(n_experiments)]

for e in range(n_experiments):
    print('Running experiment #' + str(e + 1))
    table = [[] for _ in range(num_of_subcamp)]

    environments = np.array([BudgetEnvironment(subcamp_id=1, budgets=budgets, sigma=sigma, params=[(18, 25)]),
                             BudgetEnvironment(subcamp_id=2, budgets=budgets, sigma=sigma, params=[(32, 32)]),
                             BudgetEnvironment(subcamp_id=3, budgets=budgets, sigma=sigma, params=[(8, 17)])])
    learners = np.array([GPTS_Learner(subcamp_id=1, n_arms=n_arms, arms=budgets),
                         GPTS_Learner(subcamp_id=2, n_arms=n_arms, arms=budgets),
                         GPTS_Learner(subcamp_id=3, n_arms=n_arms, arms=budgets)])

    for t in range(T):

        knapsack_table = generate_table(learners)
        best_alloc, best_value = knapsack(knapsack_table, budgets, daily_budget)

        for entry in best_alloc:
            c = entry.subcampaign
            reward = environments[c].round(entry.budget_arm)
            learners[c].update(entry.budget_arm, reward)
            rewards_per_experiment[e][t] += reward
            normalized_rewards_per_experiment[e][t] += environments[c].phases[0][entry.budget_arm]

rewards = np.mean(rewards_per_experiment, axis=0)
normalized_rewards = np.mean(normalized_rewards_per_experiment, axis=0)

# Optimum
knapsack_table = curve_table(environments)
opt_alloc, opt_value = knapsack(knapsack_table, budgets, daily_budget)

plt.figure(0)
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(np.full(T, opt_value), '--k')
plt.plot(normalized_rewards, c='g')
plt.title('Rewards')
plt.legend(["Optimum", "GPTS"])
plt.show()

plt.figure(1)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(opt_value - normalized_rewards), 'r')
plt.title('Cumulative regret')
plt.legend(["GPTS"])
plt.show()