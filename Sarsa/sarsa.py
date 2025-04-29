import numpy as np

class Sarsa:
    def __init__(self, n_states, n_actions=4, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.Q_table = np.zeros([n_states, n_actions])  # initialize Q(s, a)
        self.n_actions = n_actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # epsilon-greedy

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_actions)]
        for i in range(self.n_actions):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, state, action, reward, next_state, next_action):
        td_error = reward + self.gamma * self.Q_table[next_state, next_action] - self.Q_table[state, action]
        self.Q_table[state, action] += self.alpha * td_error