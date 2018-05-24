import numpy as np
from scipy.stats import beta


class Epsilon:
    def __init__(self, epsilon=0.1):
        self.name = 'Epsilon Greedy'
        self.epsilon = epsilon

    def evaluate(self, node_list):
        means = np.array([node.total_reward/node.visit_count+np.random.uniform(low=0.01, high = 0.05) for
                          node in node_list])
        random = np.random.uniform(0, 1)
        if random < self.epsilon:
            return np.random.choice(node_list)
        else:
            return node_list[np.argmax(means)]

    def best(self, node_list):
        means = np.array([node.total_reward/node.visit_count+np.random.uniform(low=0.01, high = 0.05) for
                          node in node_list])
        return node_list[np.argmax(means)]


class UCB:
    def __init__(self, explore_param=0.7):
        self.name = 'UCB'
        self.c = explore_param

    def evaluate(self, node_list):
        total_visits = np.sum([node.visit_count for node in node_list])
        means = np.array([node.total_reward/node.visit_count+np.random.uniform(low=0.01, high = 0.05) for node in node_list])
        cis = np.array([np.sqrt(2*np.log(total_visits)/node.visit_count) for node in node_list])
        uct_values = means + self.c * cis
        max_value = np.argmax(uct_values)
        return node_list[max_value]

    def best(self, node_list):
        means = np.array([node.total_reward/node.visit_count+np.random.uniform(low=0.01, high=0.05) for
                          node in node_list])
        return node_list[np.argmax(means)]


class Thompson:
    def __init__(self):
        self.name = 'Thompson Sampling'

    def evaluate(self, node_list):
        samples = [beta.rvs(a=node.win_count + 1, b=node.visit_count - node.win_count + 1, size=1)[0] for node in
                   node_list]
        return node_list[np.argmax(samples)]

    def best(self, node_list):
        means = [node.win_count/(node.win_count+(node.visit_count-node.win_count)) for node in node_list]
        return node_list[np.argmax(means)]