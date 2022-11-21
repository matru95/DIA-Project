import numpy as np
import math

class KS_Element:
    def __init__(self, subcampaign, budget_arm, budget, value):
        self.subcampaign = subcampaign
        self.budget_arm = budget_arm
        self.budget = budget
        self.value = value

def knapsack(input_table, budgets, max_budget):
    
    table_size = len(input_table)
    budgets_size = len(budgets)
    table = [[] for _ in range(table_size)]

    for i in range(table_size):
        for j in range(budgets_size):
            table[i].append([KS_Element(i, j, budgets[j], input_table[i][j])])
    result = [[] for _ in range(budgets_size)]

    entries = [set() for _ in range(budgets_size)]
    b1 = 0
    while budgets[b1] * 2 <= np.max(budgets):
        for b2 in range(b1, budgets_size):
            summed_b = budgets[b1] + budgets[b2]
            for b in budgets:
                if math.isclose(summed_b, b, rel_tol=1e-5):
                    buds = np.where(budgets == b)[0][0]
                    entries[buds].add((b1, b2))
                    entries[buds].add((b2, b1))
        b1 += 1
    entries = list(map(lambda x: list(x), entries))

    for i in range(1, len(table)):
        prev = table[i-1]
        curr = table[i]
        for budget in range(budgets_size):
            curr_entries = entries[budget]
            total = []
            for p in curr_entries:
                v1 = np.sum(list(map(lambda x: x.value, prev[p[0]])))
                v2 = np.sum(list(map(lambda x: x.value, curr[p[1]])))
                total.append(v1 + v2)
            best_entry = np.argmax(total)
            result[budget] = prev[curr_entries[best_entry][0]].copy() + curr[curr_entries[best_entry][1]].copy()
        table[i] = result.copy()

    i = 0
    for k in range(budgets_size):
        if budgets[k] <= max_budget:
            i = k
    
    result = result[0:i]
    best_alloc = result[np.argmax(np.sum(np.vectorize(lambda x: x.value)(result), axis=1))]

    best_alloc = list(filter(lambda x: x.budget > 0.0, best_alloc))
    best_value = np.sum(np.vectorize(lambda x: x.value)(best_alloc))
    return best_alloc, best_value