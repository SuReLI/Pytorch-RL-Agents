import os
import datetime
import random
import gym
import numpy as np


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def convert_name(title):
    date = '_'.join(title.split('_')[1:3])
    return datetime.datetime.strptime(date, '%Y-%m-%d_%H-%M-%S')


def get_latest_dir(folder):
    dirs = os.listdir(folder)
    dirs.sort(key=convert_name)
    if len(dirs) > 0:
        return dirs[-1]
    else:
        return ''
