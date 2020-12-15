import numpy as np
import matplotlib.pyplot as plt
from PriceEnvironment import *
from TS_Learner_Context import *

def most_common(lst):
    return max(set(lst), key=lst.count)

x = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70]
y1 = [0.4, 0.45, 0.55, 0.65, 0.5, 0.35, 0.25, 0.1, 0.08, 0.05, 0.02, 0.001]
y2 = [0.3, 0.45, 0.55, 0.65, 0.69, 0.72, 0.59, 0.35, 0.27, 0.13, 0.04, 0.001]
y3 = [0.1, 0.17, 0.24, 0.33, 0.46, 0.57, 0.63, 0.58, 0.3, 0.1, 0.02, 0.001]
p = [0.2, 0.3, 0.5]

n_arms = 7
prices = np.linspace(min(x), max(x), num=n_arms, endpoint=True)
num_of_subcamp = 3
T = 366
n_experiments = 100

base_env = np.array([PriceEnvironment(subcamp_id=1, n_arms=n_arms, arms=prices, x=x, y=y1, weight=p[0]),
                    PriceEnvironment(subcamp_id=2, n_arms=n_arms, arms=prices, x=x, y=y2, weight=p[1]),
                    PriceEnvironment(subcamp_id=3, n_arms=n_arms, arms=prices, x=x, y=y3, weight=p[2])])

base_env[0].plot_curve('Young', prices)
base_env[1].plot_curve('Adult', prices)
base_env[2].plot_curve('Business', prices)

y12 = list(map(lambda x, y: (x*p[0]+y*p[1]), y1, y2))
y13 = list(map(lambda x, y: (x*p[0]+y*p[2]), y1, y3))
y23 = list(map(lambda x, y: (x*p[1]+y*p[2]), y2, y3))
y123 = list(map(lambda x, y, z: (x*p[0]+y*p[1]+z*p[2]), y1, y2, y3))

aggr12_env = PriceEnvironment(subcamp_id=12, n_arms=n_arms, arms=prices, x=x, y=y12, weight=p[0]+p[1])
aggr13_env = PriceEnvironment(subcamp_id=13, n_arms=n_arms, arms=prices, x=x, y=y13, weight=p[0]+p[2])
aggr23_env = PriceEnvironment(subcamp_id=23, n_arms=n_arms, arms=prices, x=x, y=y23, weight=p[1]+p[2])
aggr123_env = PriceEnvironment(subcamp_id=123, n_arms=n_arms, arms=prices, x=x, y=y123, weight=1)

environments = [base_env[0], base_env[1], base_env[2], aggr12_env, aggr13_env, aggr23_env, aggr123_env]

contexts = [[base_env[0], base_env[1], base_env[2]],
            [aggr12_env, base_env[2]],
            [aggr13_env, base_env[1]],
            [aggr23_env, base_env[0]],
            [aggr123_env]]

optimums = []
for cont in contexts:
    opt = 0
    for env in cont:
        opt += env.optimum * env.weight
    optimums.append(opt)

rewards_per_experiment = [[] for _ in range(len(contexts))]

for i,c in enumerate(contexts):
    for env in c:
        rewards_per_experiment[i].append([[]for _ in range(n_experiments)])

best_context_per_experiment = [[]for _ in range(n_experiments)]

#context_lb = [[] for _ in range(n_experiments)]

for e in range(n_experiments):
    print('Running experiment #' + str(e+1))

    disaggr_learners = np.array([TS_Learner_Context(subcamp_id = 1, n_arms=n_arms, arms=prices),
                                 TS_Learner_Context(subcamp_id = 2, n_arms=n_arms, arms=prices),
                                 TS_Learner_Context(subcamp_id = 3, n_arms=n_arms, arms=prices)])
    aggr12_learner = TS_Learner_Context(subcamp_id = 12, n_arms=n_arms, arms=prices)
    aggr13_learner = TS_Learner_Context(subcamp_id = 13, n_arms=n_arms, arms=prices)
    aggr23_learner = TS_Learner_Context(subcamp_id = 23, n_arms=n_arms, arms=prices)
    aggr123_learner = TS_Learner_Context(subcamp_id = 123, n_arms=n_arms, arms=prices)

    learners = [disaggr_learners[0], disaggr_learners[1], disaggr_learners[2], 
                aggr12_learner, aggr13_learner, aggr23_learner, aggr123_learner]

    learners_dict = {1:disaggr_learners[0], 2:disaggr_learners[1], 3:disaggr_learners[2], 
                12:aggr12_learner, 13:aggr13_learner, 23:aggr23_learner, 123:aggr123_learner}
    
    n = 7
    confidence = 0.9
    best_context = []
    #Context 4 is the starting context for our algorithm
    best_context.append(4)

    for t in range(T):

        if (t % n == 0) and not(t==0):

            context_bound = [0 for _ in range(len(contexts))]
            for i,cont in enumerate(contexts):
                for j,env in enumerate(cont):
                    l = learners_dict[env.subcamp_id]
                    ok_arms = list(filter(l.is_pulled, range(n_arms)))
                    arms_means = list(map(lambda x:(x, l.beta_mean(x)), ok_arms))
                    best_arm, x_mean = max(arms_means, key=lambda item:item[1])
                    num_of_pulls = l.number_of_pulls[best_arm]
                    reward_lb = x_mean - np.sqrt(-1*np.log(confidence)/(2*num_of_pulls))
                    context_bound[i] += reward_lb * env.weight
            best_context.append(np.argmax(context_bound))
            
            #if e < n_experiments:
            #    context_lb[e].append(context_bound)

        for i, learner in enumerate(learners):
            pulled_arm = learner.pull_arm()
            reward = environments[i].round(pulled_arm)
            learner.update(pulled_arm, reward)

    for i,cont in enumerate(contexts):
        for j,env in enumerate(cont):
            rewards_per_experiment[i][j][e].append(learners_dict[env.subcamp_id].collected_rewards * env.weight)
    
    best_context_per_experiment[e] = best_context

rewards = []
regrets = []
for i,cont in enumerate(contexts):
    rewards.append(np.sum(np.mean(rewards_per_experiment[i], axis=1), axis=0)[0])
    title = 'Context:'
    for e in cont:
        title += ' {' + str(e.subcamp_id) + '}'

    plt.figure(2*i + 2)
    plt.title(title)
    plt.ylabel("Reward")
    plt.xlabel("t")
    plt.plot(rewards[i], 'b')
    plt.plot(np.full(T, optimums[i]), '--r')
    plt.legend(["TS_Learner", "Optimum"])
    plt.show()

    regrets.append(np.cumsum(np.full(T, optimums[i]) - rewards[i], axis=0))
    plt.figure(2*i + 1 + 2)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.plot(regrets[i], 'g')
    plt.show()

final_contexts = []
cumulative_rewards = []
cumulative_regrets = []
for k in range(int(T/n)):
    context = most_common([item[k] for item in best_context_per_experiment])
    final_contexts.append(context)
    cumulative_rewards.append(rewards[context][k*7:(k+1)*7])
    cumulative_regrets.append(regrets[context][k*7:(k+1)*7])

cumulative_rewards = [item for sublist in cumulative_rewards for item in sublist]
cumulative_regrets = [item for sublist in cumulative_regrets for item in sublist]

plt.figure(0)
plt.title('Final Reward')
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(cumulative_rewards, 'b')
plt.plot(np.full(T, optimums[0]), '--r')
plt.legend(["TS_Learner", "Optimum"])
plt.show()

plt.figure(1)
plt.title('Final Regret')
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(cumulative_regrets, 'g')
plt.show()

'''
aaa = 0
for r in context_lb:
    plt.figure(aaa+30)
    plt.title("Context's Rewards-per-week Lowerbounds")
    plt.xlabel("t")
    plt.ylabel("Rewards LB")
    plt.plot(list(item[0] for item in r), 'r')
    plt.plot(list(item[1] for item in r), 'g')
    plt.plot(list(item[2] for item in r), 'b')
    plt.plot(list(item[3] for item in r), 'c')
    plt.plot(list(item[4] for item in r), 'm')
    s1 = str(f"{{a}}, {{b}}, {{c}}")
    s2 = str(f"{{a, b}}, {{c}}")
    s3 = str(f"{{a, c}}, {{b}}")
    s4 = str(f"{{a}}, {{b, c}}")
    s5 = str(f"{{a, b, c}}")
    plt.legend([s1, s2, s3, s4, s5])
    plt.savefig('reward_LB_' + str(aaa))
    plt.show()
    aaa +=1
'''