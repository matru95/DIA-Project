import numpy as np
import matplotlib.pyplot as plt

from shared.PriceEnvironment import *
from shared.TS_Learner import *
from shared.BudgetEnvironment import *
from shared.GPTS_Learner import *
from shared.Knapsack import *

def generate_table(learners, value_per_sc):
    learners = np.atleast_1d(learners)
    table = np.array(list(map(lambda x: x.draw_samples()*value_per_sc[x.subcamp_id-1], learners)))
    table[:, 0] = 0
    return table

def curve_table(envs):
    envs = np.atleast_1d(envs)
    return np.array(list(map(lambda x: x.phases[0], envs)))

T = int(366/3)
n_experiments = 30
num_of_subcamp = 3

#PRICING
x = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70]
y1 = [0.4, 0.45, 0.55, 0.65, 0.5, 0.35, 0.25, 0.1, 0.08, 0.05, 0.02, 0.001]
y2 = [0.3, 0.45, 0.55, 0.65, 0.69, 0.72, 0.59, 0.35, 0.27, 0.13, 0.04, 0.001]
y3 = [0.1, 0.17, 0.24, 0.33, 0.46, 0.57, 0.63, 0.58, 0.3, 0.1, 0.02, 0.001]

p = [0.2, 0.4, 0.4]

n_price_arms = 4
prices = np.linspace(min(x), max(x), num=n_price_arms, endpoint=True)

#ADVERTISING
n_budget_arms = 7
min_budget = 0.0
max_budget = 50.0
budgets = np.linspace(min_budget, max_budget, n_budget_arms)
sigma = 1
daily_budget = 50.0

#Rewards per Pricing
rewards_per_experiment_pricing = [[[] for _ in range(n_experiments)] for _ in range(num_of_subcamp)]
#Rewards per Advertising
rewards_per_experiment = [np.zeros(T) for _ in range(n_experiments)]
normalized_rewards_per_experiment = [np.zeros(T) for _ in range(n_experiments)]

#Pricing Environments
price_env = np.array([PriceEnvironment(subcamp_id=1, n_arms=n_price_arms, arms=prices, x=x, y=y1),
                      PriceEnvironment(subcamp_id=2, n_arms=n_price_arms, arms=prices, x=x, y=y2),
                      PriceEnvironment(subcamp_id=3, n_arms=n_price_arms, arms=prices, x=x, y=y3)])
                
#Advertising Environments
adv_env = np.array([BudgetEnvironment(subcamp_id = 1, budgets=budgets, sigma=sigma, params=[(18, 25)]),
                    BudgetEnvironment(subcamp_id = 2, budgets=budgets, sigma=sigma, params=[(32, 32)]),
                    BudgetEnvironment(subcamp_id = 3, budgets=budgets, sigma=sigma, params=[(8, 17)])])

for e in range(n_experiments):
    print('Running experiment #' + str(e+1))

    pricing_learners = np.array([TS_Learner(subcamp_id = 1, n_arms=n_price_arms, arms=prices),
                                TS_Learner(subcamp_id = 2, n_arms=n_price_arms, arms=prices),
                                TS_Learner(subcamp_id = 3, n_arms=n_price_arms, arms=prices)])
    
    knapsack_table = [[] for _ in range(num_of_subcamp)]

    adv_learners = np.array([GPTS_Learner(subcamp_id = 1, n_arms=n_budget_arms, arms=budgets),
                            GPTS_Learner(subcamp_id = 2, n_arms=n_budget_arms, arms=budgets),
                            GPTS_Learner(subcamp_id = 3, n_arms=n_budget_arms, arms=budgets)])

    for t in range(T):

        if t % 10 == 0: print('Time: ' + str(t))
        
        #PRICING
        value_per_sc = []
        for i, learner in enumerate(pricing_learners):
            pulled_arm = learner.pull_arm()
            reward = price_env[i].round(pulled_arm)
            learner.update(pulled_arm, reward)
            value_per_sc.append(learner.beta_mean(pulled_arm) * prices[pulled_arm])
        #ADVERTISING
        knapsack_table = generate_table(adv_learners, value_per_sc)
        best_alloc, best_value = knapsack(knapsack_table, budgets, daily_budget)

        for entry in best_alloc:
            c = entry.subcampaign
            reward = adv_env[c].round(entry.budget_arm)
            adv_learners[c].update(entry.budget_arm, reward)
            rewards_per_experiment[e][t] += reward
            normalized_rewards_per_experiment[e][t] += adv_env[c].phases[0][entry.budget_arm]
    
    #PRICING Reward
    for i, learner in enumerate(pricing_learners):
        rewards_per_experiment_pricing[i][e].append(learner.collected_rewards*p[i])

rewards_pricing = np.sum(np.mean(rewards_per_experiment_pricing, axis=1), axis=0)[0]

#ADVERTISING Reward
rewards_adv = np.mean(rewards_per_experiment, axis=0)
normalized_rewards_adv = np.mean(normalized_rewards_per_experiment, axis=0)

#Pricing Optimum
opt_disaggr = (price_env[0].optimum*p[0] + price_env[1].optimum*p[1] + price_env[2].optimum*p[2])

#Advertising Optimum
knapsack_table = curve_table(adv_env)
opt_alloc, opt_value = knapsack(knapsack_table, budgets, daily_budget)

plt.figure(0)
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(np.full(T, opt_disaggr), '--k')
plt.plot(rewards_pricing, c='c')
plt.title('Price Rewards - Disaggregate Case')
plt.legend(["Optimum", "TS"])
plt.show()

plt.figure(1)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(opt_disaggr - rewards_pricing), 'm')
plt.title('Price Regret - Disaggregate Case')
plt.legend(["TS"])
plt.show()

plt.figure(0)
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(np.full(T, opt_value), '--k')
plt.plot(normalized_rewards_adv, c='g')
plt.title('Budget Rewards - Disaggregate Case')
plt.legend(["Optimum", "GPTS"])
plt.show()

plt.figure(1)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(opt_value - normalized_rewards_adv), 'r')
plt.title('Budget Regret - Disaggregate Case')
plt.legend(["GPTS"])
plt.show()