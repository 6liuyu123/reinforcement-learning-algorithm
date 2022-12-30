import numpy as np
import time

MAZE_W = 4

class QLearning(object):
    def __init__(self, state_size, action_size, learning_rate = 0.01, gamma = 0.9, epsilon = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.QTable = np.zeros((state_size, action_size))
        self.train_times = 100
        self.episode = 100

    def state_transition(self, state):
        return int(((state[0] - 5) * MAZE_W + (state[1] - 5)) / 40)

    def get_action(self, state):
        sand = np.random.uniform(0, 1)
        if sand <= self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            Q_max = np.max(self.QTable[state])
            action_list = np.where(self.QTable[state,:] == Q_max)[0]
            action = np.random.choice(action_list)
        return action
    
    def update_QTable(self, state, action, next_state, reward):
        if next_state == 'terminal':
            Q_next = 0
        else:
            Q_next = np.max(self.QTable[self.state_transition(next_state),:])
        self.QTable[self.state_transition(state), action] += self.learning_rate * (reward + self.gamma * Q_next - self.QTable[self.state_transition(state), action])

    def train(self, env, agent):
        success_time = 0
        for i in range(self.train_times):
            print(success_time)
            for j in range(self.episode):
                state = env.reset()
                while True:
                    action = self.get_action(self.state_transition(state))
                    next_state, reward, done = env.step(action)
                    agent.update_QTable(state, action, next_state, reward)
                    state = next_state
                    if done:
                        success_time += 1
                        break
                    if next_state == 'terminal':
                        break
