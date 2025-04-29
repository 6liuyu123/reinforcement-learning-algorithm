import gymnasium as gym
import matplotlib.pyplot as plt
from sarsa import Sarsa

def main():
    env = gym.make("Taxi-v3")
    max_steps = 100
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    n_episodes = 20000
    agent = Sarsa(
        n_states=n_states,
        n_actions=n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
    )
    return_list = []
    for episode in range(n_episodes):
        state = env.reset()[0]
        done = False
        for i in range(max_steps):
            state = env.reset()[0]
            episode_return = 0
            action = agent.take_action(state)
            done = False
            while not done:
                next_state, reward, done, truncated, info = env.step(action) # observation, reward, terminated, truncated, info
                next_action = agent.take_action(next_state)
                episode_return += reward
                agent.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
            return_list.append(episode_return)
            if done:
                break
    env.close()
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Sarsa on {}'.format('Taxi-v3'))
    plt.show()

if __name__ == "__main__":
    main()