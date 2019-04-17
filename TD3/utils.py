"""Shared classes and parameters for the benchmark on several gym games"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# One class for parameters

class Config:
    GAME = "Pendulum-v0"

    MEMORY_CAPACITY = 1000000
    BATCH_SIZE = 100
    GAMMA = 0.99
    LEARNING_RATE_CRITIC = 0.001
    LEARNING_RATE_ACTOR = 0.001
    TAU = 0.005

    MAX_EPISODES = 1000
    MAX_STEPS = 200
    EPSILON = 0.001


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_critic(nn.Module):

    def __init__(self, input_size):
        super(DQN_critic, self).__init__()

        self.hidden1 = nn.Linear(input_size, 8)
        self.hidden2 = nn.Linear(8, 8)
        self.hidden3 = nn.Linear(8, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        return self.output(x.view(x.size(0), -1))


class DQN_actor(nn.Module):

    def __init__(self, state_size, action_size, low_bound, high_bound):
        super(DQN_actor, self).__init__()

        self.hidden1 = nn.Linear(state_size, 8)
        self.hidden2 = nn.Linear(8, 8)
        self.hidden3 = nn.Linear(8, 8)
        self.output = nn.Linear(8, action_size)
        self.low_bound = low_bound
        self.high_bound = high_bound

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.output(x)
        x = (torch.sigmoid(x) * (self.high_bound - self.low_bound)) + self.low_bound
        return x.view(x.size(0), -1)


# Soft target update function
def update_targets(target, original, tau):
    """Weighted average update of the target network and original network
        Inputs: target actor(critic) and original actor(critic)"""

    for targetParam, orgParam in zip(target.parameters(), original.parameters()):
        targetParam.data.copy_((1 - tau)*targetParam.data + tau*orgParam.data)
