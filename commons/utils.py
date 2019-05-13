import os
import datetime
import random


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
