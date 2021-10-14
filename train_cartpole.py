# Cartpole training implementation is based on https://gist.github.com/Pocuston/13f1a7786648e1e2ff95bfad02a51521
# which is a slight modification to http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# where we simplify the problem by not using the frames but the action space itself.

import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

steps_done = 0
episode_durations = []

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self, hidden_layer):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, hidden_layer)
        self.l2 = nn.Linear(hidden_layer, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

def select_action(state, model, eps_start, eps_end, eps_decay):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        return model(state.type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])


def run_episode(e, environment, model, eps_start, eps_end, eps_decay, gamma, batch_size, memory, optimizer):
    state = environment.reset()
    steps = 0
    while True:
        environment.render()
        action = select_action(FloatTensor([state]), model, eps_start, eps_end, eps_decay)
        next_state, reward, done, _ = environment.step(action[0, 0].item())

        # negative reward when attempt ends
        if done:
            reward = -1

        memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward])))

        learn(memory, batch_size, model, gamma, optimizer)

        state = next_state
        steps += 1

        if done:
            print("{2} Episode {0} finished after {1} steps"
                  .format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            episode_durations.append(steps)
            plot_durations()
            break


def learn(memory, batch_size, model, gamma, optimizer):
    if len(memory) < batch_size:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (gamma * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    expected_q_values = expected_q_values.unsqueeze(1)
    loss = F.smooth_l1_loss(current_q_values, expected_q_values)

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def train_cartpole(env, episodes, eps_start, eps_end, eps_decay, gamma,
                    learning_rate, hidden_layer, batch_size, max_size):

    model = Network(hidden_layer)

    if use_cuda:
        model.cuda()

    memory = ReplayMemory(max_size)
    optimizer = optim.Adam(model.parameters(), learning_rate)

    for e in range(episodes):
        run_episode(e, env, model, eps_start, eps_end, eps_decay, gamma, batch_size, memory, optimizer)

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()