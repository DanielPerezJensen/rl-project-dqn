# Pendulum implementation is based on https://github.com/xtma/simple-pytorch-rl/blob/master/dqn.py

import argparse
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])


class Net(nn.Module):

    def __init__(self, num_actions):
        super(Net, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.a_head = nn.Linear(100, num_actions)
        self.v_head = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        a = self.a_head(x) - self.a_head(x).mean(1, keepdim=True)
        v = self.v_head(x)
        action_scores = a + v
        return action_scores


class Memory():

    data_pointer = 0
    isfull = False

    def __init__(self, capacity):
        self.memory = np.empty(capacity, dtype=object)
        self.capacity = capacity

    def update(self, transition):
        self.memory[self.data_pointer] = transition
        self.data_pointer += 1
        if self.data_pointer == self.capacity:
            self.data_pointer = 0
            self.isfull = True

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)


class Agent():

    def __init__(self, num_actions, gamma, q_function):
        self.action_list = [(i * 4 - 2,) for i in range(num_actions)]
        self.q_function = q_function
        self.max_grad_norm = 0.5
        self.training_step = 0
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = 1
        self.eval_net, self.target_net = Net(num_actions).float(), Net(num_actions).float()
        self.memory = Memory(2000)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if np.random.random() < self.epsilon:
            action_index = np.random.randint(self.num_actions)
        else:
            probs = self.eval_net(state)
            action_index = probs.max(1)[1].item()
        return self.action_list[action_index], action_index

    def save_param(self):
        torch.save(self.eval_net.state_dict(), 'pendulum/pendulum_net_params.pkl')

    def store_transition(self, transition):
        self.memory.update(transition)

    def update(self):
        self.training_step += 1

        transitions = self.memory.sample(32)
        s = torch.tensor([t.s for t in transitions], dtype=torch.float)
        a = torch.tensor([t.a for t in transitions], dtype=torch.long).view(-1, 1)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float)

        if self.q_function == "vanilla":
            q_eval = self.eval_net(s).gather(1, a)
            with torch.no_grad():
                q_target = r + self.gamma * self.target_net(s_).max(1, keepdim=True)[0]

        elif self.q_function == "double":
            with torch.no_grad():
                a_ = self.eval_net(s_).max(1, keepdim=True)[1]
                q_target = r + self.gamma * self.target_net(s_).gather(1, a_)
            q_eval = self.eval_net(s).gather(1, a)

        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(q_eval, q_target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.training_step % 200 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.epsilon = max(self.epsilon * 0.999, 0.01)

        return q_eval.mean().item()


def train_pendulum(env, gamma, num_actions, render, log_interval, q_function):

    env.seed(SEED)

    agent = Agent(num_actions, gamma, q_function)

    training_records = []
    running_reward, running_q = -1000, 0
    for i_ep in range(100):
        score = 0
        state = env.reset()

        for t in range(200):
            action, action_index = agent.select_action(state)
            state_, reward, done, _ = env.step(action)
            score += reward
            if render:
                env.render()
            agent.store_transition(Transition(state, action_index, (reward + 8) / 8, state_))
            state = state_
            if agent.memory.isfull:
                q = agent.update()
                running_q = 0.99 * running_q + 0.01 * q

        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward))

        if i_ep % log_interval == 0:
            print('Ep {}\tAverage score: {:.2f}\tAverage Q: {:.2f}'.format(
                i_ep, running_reward, running_q))
        if running_reward > -200:
            print("Solved! Running reward is now {}!".format(running_reward))
            env.close()
            agent.save_param()
            with open('pendulum/pendulum_training_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)
            break

    env.close()

    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('DQN')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig(f"pendulum/dqn_{q_function}.png")
    plt.show()