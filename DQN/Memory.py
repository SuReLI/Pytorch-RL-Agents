import random
from collections import deque

# from parameters import Parameters

from utils import Transition

class ReplayMemory:
    """A class to store the transitions (s, a, r, s') and to sample them"""

    def __init__(self, capacity, n_step, gamma):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.memory = []
        self.n_step_memory = deque()
        self.position = 0

    # Saving a transition tuple
    def push(self, *args):
        self.n_step_memory.append(args)

        if len(self.n_step_memory) >= self.n_step:
            s_mem, a_mem, R, si_ = self.n_step_memory.popleft()
            for i in range(self.n_step-1):
                si, ai, ri, si_ = self.n_step_memory[i]
                if si is None:
                    break
                R += ri * self.gamma ** (i+1)

            if len(self.memory) < self.capacity:
                self.memory.append(Transition(s_mem, a_mem, R, si_))
            else:
                self.memory[self.position] = Transition(s_mem, a_mem, R, si_)
                self.position = (self.position + 1) % self.capacity


    # Sample a random number according to batch size
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
