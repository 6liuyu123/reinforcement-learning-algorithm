import gym
import numpy as np
import random
import config

from DQN import DQN

def main():
    env = gym.make("CartPole-v1")
    config.state_size = env.observation_space.shape[0]
    config.action_size = env.action_space.n 
    agent = DQN()
    for i in range(config.episode):
        print('------------Episode %s------------' % i)
        s = env.reset()[0]
        episode_reward_sum = 0
        while True:
            env.render()
            a = agent.choose_action(s)
            step_msg= env.step(a)
            s_ = step_msg[0]
            r = step_msg[1]
            done = step_msg[2]
            agent.store_transition(s, a, r, s_)
            episode_reward_sum += r
            s = s_
            if agent.memory_counter > config.memory_capacity:
                agent.learn()
            if done:
                print('------------Episode %s\'s reward sum is %s------------' % (i, episode_reward_sum))
                break
    env.close()

if __name__ == "__main__":
    main()