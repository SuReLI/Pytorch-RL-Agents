import os
import datetime
import random
import gym
import numpy as np
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


class NStepsReplayMemory(ReplayMemory):

    def __init__(self, capacity, n_step, gamma):
        super().__init__(capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.nstep_memory = deque()

    def _process_n_step_memory(self):
        s_mem, a_mem, R, si_, done = self.nstep_memory.popleft()
        if not done:
            for i in range(self.n_step-1):
                si, ai, ri, si_, done = self.nstep_memory[i]
                R += ri * self.gamma ** (i+1)
                if done:
                    break

        return [s_mem, a_mem, R, si_, done]

    def push(self, *transition):
        self.nstep_memory.append(transition)
        while len(self.nstep_memory) >= self.n_step or (self.nstep_memory and self.nstep_memory[-1][4]):
            nstep_transition = self._process_n_step_memory()
            super().push(*nstep_transition)


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        # Discrete envs
        if hasattr(self.action_space, 'n'):
            return action

        assert (-1 <= action).all() and (action <= 1).all(), "Action not valid"

        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        # Discrete envs
        if hasattr(self.action_space, 'n'):
            return action

        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


def get_epsilon_threshold(episodes, params):
    """Returns the epsilon value (for exploration) following an exponential decreasing. """
    return params['EPSILON_END'] + (params['EPSILON_START'] - params['EPSILON_END']) * \
        math.exp(-1 * episodes / params['EPSILON_DECAY'])


def str_to_list(string):
    return list(map(float, string[1:-1].split(', ')))


def convert_name(title):
    date = title.split('_', 1)[1]
    return datetime.datetime.strptime(date, '%Y-%m-%d_%H-%M-%S')


def is_valid(title):
    try:
        convert_name(title)
        return True
    except (IndexError, ValueError):
        return False


def get_latest_dir(folder):
    try:
        dirs = os.listdir(folder)
    except FileNotFoundError:
        raise FileNotFoundError(f"No expe saved in the folder {folder} !") from None

    dirs = list(filter(is_valid, dirs))
    dirs.sort(key=convert_name)
    if len(dirs) > 0:
        return os.path.join(folder, dirs[-1])
    else:
        raise FileNotFoundError(f"No valid file in the folder {folder} !")


def get_current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def sample_to_str(transition):
    s, a, r, s_, d = transition
    data = [list(s), list(a), r, list(s_), 1-int(d)]
    return ' ; '.join(map(str, data)) + '\n'


def write_transitions(s, a, r, s_, d, file_name='transitions.csv'):

    data = sample_to_str((s, a, r, s_, d))
    with open(file_name, 'a') as file:
        file.write(data)
