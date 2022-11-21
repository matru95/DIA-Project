import numpy as np
import matplotlib.pyplot as plt

from shared.NS_BudgetEnvironment import *
from shared.TS_Learner import *
from shared.SWTS_Learner import *
from shared.Knapsack import *

def generate_table(learners, env):
    learners = np.atleast_1d(learners)
    table = np.array(list(map(lambda x: x.draw_samples()*env[x.subcamp_id-1].max_clicks, learners)))
    table[:, 0] = 0
    return table

def curve_table(envs, phase_idx):
    envs = np.atleast_1d(envs)
    return np.array(list(map(lambda x: x.phases[phase_idx], envs)))

def plot_curve_per_phase(i, p1, p2, p3):
        plt.figure(0)
        buds = np.linspace(0.0, 50.0, 100)
        plt.ylabel("Number of Clicks")
        plt.xlabel("budget")
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        plt.title('Phase: ' + str(i+1))
        plt.plot(buds, build_real_curve(buds, p1), 'r')
        plt.plot(buds, build_real_curve(buds, p2), 'b')
        plt.plot(buds, build_real_curve(buds, p3), 'g')
        plt.legend(["Young","Adult","Business"])
        plt.show()

n_arms = 11
min_budget = 0.0
max_budget = 50.0
budgets = np.linspace(min_budget, max_budget, n_arms)
sigma = 1
daily_budget = [55.0, 60.0, 70.0]
num_of_subcamp = 3
T = 366
n_experiments = 10
n_phases = 3
window_size = int(np.sqrt(T))
phases_len_float = T / n_phases
phases_len = int(phases_len_float)

phases = [[(18, 25),(15, 45),(20, 41)],  #YOUNG
          [(32, 32),(40, 40),(33, 34)],  #ADULTS
          [(8, 17),(29, 50),(22, 30)]]   #BUSINESS

for i in range(n_phases):
    plot_curve_per_phase(i, phases[0][i], phases[1][i], phases[2][i])

ts_rewards_per_experiment = [np.zeros(T) for _ in range(n_experiments)]
ts_normalized_rewards_per_experiment = [np.zeros(T) for _ in range(n_experiments)]
swts_rewards_per_experiment = [np.zeros(T) for _ in range(n_experiments)]
swts_normalized_rewards_per_experiment = [np.zeros(T) for _ in range(n_experiments)]

ts_environments = np.array([NS_BudgetEnvironment(subcamp_id = 1, horizon=T, budgets=budgets, sigma=sigma, params=phases[0]),
                            NS_BudgetEnvironment(subcamp_id = 2, horizon=T, budgets=budgets, sigma=sigma, params=phases[1]),
                            NS_BudgetEnvironment(subcamp_id = 3, horizon=T, budgets=budgets, sigma=sigma, params=phases[2])])  
swts_environments = np.array([NS_BudgetEnvironment(subcamp_id = 1, horizon=T, budgets=budgets, sigma=sigma, params=phases[0]),
                              NS_BudgetEnvironment(subcamp_id = 2, horizon=T, budgets=budgets, sigma=sigma, params=phases[1]),
                              NS_BudgetEnvironment(subcamp_id = 3, horizon=T, budgets=budgets, sigma=sigma, params=phases[2])])  

for e in range(n_experiments):
    print('Running experiment #' + str(e+1))

    ts_knapsack_table = [[] for _ in range(num_of_subcamp)]
    ts_learners = np.array([TS_Learner(subcamp_id = 1, n_arms=n_arms, arms=budgets),
                            TS_Learner(subcamp_id = 2, n_arms=n_arms, arms=budgets),
                            TS_Learner(subcamp_id = 3, n_arms=n_arms, arms=budgets)])

    swts_knapsack_table = [[] for _ in range(num_of_subcamp)]
    swts_learners = np.array([SWTS_Learner(subcamp_id = 1, n_arms=n_arms, arms=budgets, window_size=window_size),
                              SWTS_Learner(subcamp_id = 2, n_arms=n_arms, arms=budgets, window_size=window_size),
                              SWTS_Learner(subcamp_id = 3, n_arms=n_arms, arms=budgets, window_size=window_size)])
    
    for t in range(T):
        curr_phase = int(t / phases_len_float)
        
        ts_knapsack_table = generate_table(ts_learners, ts_environments)
        ts_best_alloc, ts_best_value = knapsack(ts_knapsack_table, budgets, daily_budget[curr_phase])
        for entry in ts_best_alloc:
            c = entry.subcampaign
            ts_reward = ts_environments[c].round(entry.budget_arm, t)
            ts_learners[c].update(entry.budget_arm, ts_reward)
            ts_rewards_per_experiment[e][t] += ts_reward
            ts_normalized_rewards_per_experiment[e][t] += ts_environments[c].phases[curr_phase][entry.budget_arm]
        
        swts_knapsack_table = generate_table(swts_learners, swts_environments)
        swts_best_alloc, swts_best_value = knapsack(swts_knapsack_table, budgets, daily_budget[curr_phase])
        for entry in swts_best_alloc:
            c = entry.subcampaign
            swts_reward = swts_environments[c].round(entry.budget_arm, t)
            swts_learners[c].update(entry.budget_arm, swts_reward)
            swts_rewards_per_experiment[e][t] += swts_reward
            swts_normalized_rewards_per_experiment[e][t] += swts_environments[c].phases[curr_phase][entry.budget_arm]

    for e in ts_environments:
        e.reset_time()
    for e in swts_environments:
        e.reset_time()

ts_rewards = np.mean(ts_rewards_per_experiment, axis=0)
ts_normalized_rewards = np.mean(ts_normalized_rewards_per_experiment, axis=0)
swts_rewards = np.mean(swts_rewards_per_experiment, axis=0)
swts_normalized_rewards = np.mean(swts_normalized_rewards_per_experiment, axis=0)

ts_instantaneous_regret = np.zeros(T)
swts_instantaneous_regret = np.zeros(T)


optimum_per_round = np.zeros(T)
for i in range(n_phases):
    # Optimum knapsack per phase
    knapsack_table = curve_table(ts_environments, i)
    opt_alloc, opt_value = knapsack(knapsack_table, budgets, daily_budget[i])
    # For each phase calculate the optimal arm per round (for every t of the phase) setting it equal to the optimal for the phase
    optimum_per_round[i * phases_len:(i + 1) * phases_len] = opt_value
    # For each phase calculate the instantaneous regret for TS and SWTS: instantaneous_regret_t = opt_t - avg_regret_t
    ts_instantaneous_regret[i * phases_len:(i + 1) * phases_len] = opt_value - ts_normalized_rewards[i * phases_len:(i + 1) * phases_len]
    swts_instantaneous_regret[i * phases_len:(i + 1) * phases_len] = opt_value - swts_normalized_rewards[i * phases_len:(i + 1) * phases_len]

plt.figure(0)
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(ts_normalized_rewards, 'r')
plt.plot(swts_normalized_rewards, 'b')
plt.plot(optimum_per_round, '--k')
plt.legend(["TS", "SW-TS", "Optimum"])
plt.show()

plt.figure(1)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(ts_instantaneous_regret, axis=0), 'r')
plt.plot(np.cumsum(swts_instantaneous_regret, axis=0), 'b')
plt.legend(["TS","SW-TS"])
plt.show()