import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN_cartpole(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(n_input, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class DQN_pendulum(nn.Module):
    def __init__(self, num_actions):
        super(DQN_pendulum, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.a_head = nn.Linear(100, num_actions)
        self.v_head = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        a = self.a_head(x) - self.a_head(x).mean(1, keepdim=True)
        v = self.v_head(x)
        action_scores = a + v
        return action_scores