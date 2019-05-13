import os
import datetime
import random

import math
from collections import deque


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

    def write(self, file_name):
        data = ''.join(map(sample_to_str, self.memory))
        with open(file_name, 'w') as file:
            file.write(data)

    def __len__(self):
        return len(self.memory)


class NStepsReplayMemory:

    def __init__(self, capacity, n_step, gamma):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.memory = []
        self.n_step_memory = deque()
        self.position = 0

    def push(self, *transition):
        self.n_step_memory.append(transition)
        if len(self.n_step_memory) >= self.n_step:
            s_mem, a_mem, R, si_, done = self.n_step_memory.popleft()
            for i in range(self.n_step-1):
                si, ai, ri, si_, done = self.n_step_memory[i]
                if si is None:
                    break
                R += ri * self.gamma ** (i+1)

            if len(self.memory) < self.capacity:
                self.memory.append([s_mem, a_mem, R, si_, done])
            else:
                self.memory[self.position] = [s_mem, a_mem, R, si_, done]
                self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def get_epsilon_threshold(episodes, params):
    """Returns the epsilon value (for exploration) following an exponential decreasing. """
    return params['EPSILON_END'] + (params['EPSILON_START'] - params['EPSILON_END']) * \
        math.exp(-1 * episodes / params['EPSILON_DECAY'])


def update_targets(target, original, tau):
    for targetParam, orgParam in zip(target.parameters(), original.parameters()):
        targetParam.data.copy_((1 - tau)*targetParam.data + tau*orgParam.data)


def str_to_list(string):
    return list(map(float, string[1:-1].split(', ')))


def convert_name(title):
    date = title.split('_', 1)[1]
    return datetime.datetime.strptime(date, '%Y-%m-%d_%H-%M-%S')


def get_latest_dir(folder):
    dirs = os.listdir(folder)
    dirs.sort(key=convert_name)
    if len(dirs) > 0:
        return dirs[-1]
    else:
        return ''


def sample_to_str(transition):
    s, a, r, s_, d = transition
    data = [list(s), list(a), r, list(s_), 1-int(d)]
    return ' ; '.join(map(str, data)) + '\n'


def write_transitions(s, a, r, s_, d, file_name='transitions.csv'):

    data = sample_to_str((s, a, r, s_, d))
    with open(file_name, 'a') as file:
        file.write(data)
