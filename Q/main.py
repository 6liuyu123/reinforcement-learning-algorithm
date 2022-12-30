import numpy as np
import random

from Maze import Maze
from QLearning import QLearning

def main():
    env = Maze()
    action_size = env.n_actions
    state_size = env.n_states
    agent = QLearning(state_size, action_size, learning_rate = 0.1, gamma = 0.9, epsilon = 0.3)
    agent.train(env, agent)

if __name__ == "__main__":
    main()