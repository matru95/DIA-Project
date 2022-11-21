import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class PriceEnvironment:

    def __init__(self, subcamp_id, n_arms, arms, x, y, weight=1):
        self.subcamp_id = subcamp_id
        self.n_arms = n_arms
        interpolate = interp1d(x, y, kind='cubic')
        self.prices = interpolate(arms)
        self.optimum = max(self.prices)
        #self.plot_curve(arms)
        self.weight = weight

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.prices[pulled_arm])
        return reward

    def plot_curve(self, title, arms):
        plt.figure(0)
        plt.title(title)
        plt.ylabel("Conversion Rate")
        plt.xlabel("Price")
        plt.plot(arms, self.prices, 'b')
        plt.ylim(-0.01,1)
        plt.xlim(10,70)
        plt.show()
'''
    def plot_curve(self, arms):
        plt.figure(0)
        plt.ylabel("Conversion Rate")
        plt.xlabel("Price")
        plt.title('Price #' + str(self.subcamp_id))
        plt.ylim(-0.01,1)
        plt.xlim(10,70)
        plt.plot(arms, self.prices, 'r')
        plt.show()
'''