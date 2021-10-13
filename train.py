# Cartpole training implementation is based on https://gist.github.com/Pocuston/13f1a7786648e1e2ff95bfad02a51521
# which is a slight modification to http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# where we simplify the problem by not using the frames but the action space itself.

# Pendulum implementation is based on https://github.com/xtma/simple-pytorch-rl/blob/master/dqn.py

import gym
from gym import wrappers
import random
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import namedtuple

use_cuda = torch.cuda.is_available()
processor = 'cuda' if use_cuda else "cpu"
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def plot_durations(episode_durations):
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

def select_action_cartpole(state, model, eps_start, eps_end, eps_decay, steps_done):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        return model(state.type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])

def select_action_pendulum(state, model, num_actions, steps_done, eps_start, eps_end, eps_decay):
    action_list = [(i * 4 - 2,) for i in range(num_actions)]
    state = torch.from_numpy(state).float().unsqueeze(0).to(processor)
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        output = model(state)
        index = output.max(1)[1].item()
    else:
        index = np.random.randint(num_actions)
    return action_list[index], index

def learn(memory, batch_size, model, learning_rate, optimizer):
    if len(memory) < batch_size:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
    batch_state = torch.cat(batch_state)
    batch_action = torch.cat(batch_action)
    batch_reward = torch.cat(batch_reward)
    batch_next_state = torch.cat(batch_next_state)

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (learning_rate * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    expected_q_values = expected_q_values.unsqueeze(1)
    loss = F.smooth_l1_loss(current_q_values, expected_q_values)

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

def run_episode(e, environment, env_name, memory,
                batch_size, model, learning_rate, optimizer,
                eps_start, eps_end, eps_decay, steps_done, p_actions,
                TrainingRecord, running_reward, score):

    state = environment.reset()

    steps = 0
    results = []

    while True:
        environment.render()
        if env_name == "cartpole":
            action = select_action_cartpole(FloatTensor([state]), model, eps_start, eps_end, eps_decay, steps_done)
            next_state, reward, done, _ = environment.step(action[0, 0].item())
        if env_name == "pendulum":
            action, index = select_action_pendulum(state, model, p_actions, steps_done, eps_start, eps_end, eps_decay)
            next_state, reward, done, _ = environment.step(action)
            action = LongTensor([[index]])
            score += reward
            running_reward = running_reward * 0.9 + score * 0.1

        # negative reward when attempt ends
        if done:
            reward = -1

        memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward])))

        learn(memory, batch_size, model, learning_rate, optimizer)

        state = next_state
        steps += 1

        if env_name == "cartpole":
            if done:
                return TrainingRecord(e, steps)
        else:
            if done:
                print("Episode {}: Running reward is now {}!".format(e, running_reward))
                break

    if env_name == "pendulum":
        if running_reward > -300:
            print("Episode {}: Solved! Running reward is now {}!".format(e, running_reward))
            environment.close()
        return TrainingRecord(e, running_reward)

def train(env, env_name, model, memory, optimizer, episodes,
          eps_start, eps_end, eps_decay, batch_size, learning_rate, p_actions):


    steps_done = 0
    model = model.to(processor)
    TrainingRecord = namedtuple('TrainingRecord', ["ep", "reward"])
    running_reward = -1000
    results = []
    for e in range(episodes):
        score = 0
        result = run_episode(e, env, env_name, memory,
                batch_size, model, learning_rate, optimizer,
                eps_start, eps_end, eps_decay, steps_done, p_actions, TrainingRecord,
                running_reward, score)
        if env_name == "cartpole":
            print("{2} Episode {0} finished after {1} steps"
                  .format(e, result.reward, '\033[92m' if result.reward >= 195 else '\033[99m'))
            results.append(result)
        if env_name == "pendulum":
            results.append(result)

    plt.plot([r.ep for r in results], [r.reward for r in results])
    plt.title('DQN')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()

    print("Complete!")
    env.close()
    plt.ioff()
    plt.show()