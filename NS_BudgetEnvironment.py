from BudgetEnvironment import *

class NS_BudgetEnvironment(BudgetEnvironment):
    
    def __init__(self, subcamp_id, horizon, budgets, sigma, params):
        super(NS_BudgetEnvironment, self).__init__(subcamp_id, budgets, sigma, params)
        self.horizon = horizon
        self.t = 0
        self.max_clicks = np.max(self.phases[0])

    def round(self, pulled_arm, t):
        n_phases = len(self.phases)
        phase_size = self.horizon / n_phases
        current_phase = int(t / phase_size)
        self.max_clicks = np.max(self.phases[current_phase])
        p = self.phases[current_phase][pulled_arm]
        self.t += 1
        return np.random.binomial(1, p/self.max_clicks)

    def reset_time(self):
        self.t = 0