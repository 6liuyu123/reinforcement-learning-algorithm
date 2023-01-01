import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

from Net import Net

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((config.memory_capacity, config.state_size * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = config.learning_rate)
        self.loss_func = nn.MSELoss()
    
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < config.epsilon:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, config.action_size)
        return action
    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % config.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % config.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        sample_index = np.random.choice(config.memory_capacity, config.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:,:config.state_size])
        b_a = torch.LongTensor(b_memory[:, config.state_size: config.state_size + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, config.state_size + 1: config.state_size + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -config.state_size:])
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + config.Gamma * q_next.max(1)[0].view(config.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()