import sys
sys.path.extend(["../commons/"])
import random

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from utils import *
from networks import DQN


class Agent:
    def __init__(self, state_size, action_size, device, config):
        self.device = device
        self.config = config

        self.nn = DQN(state_size, action_size).to(self.device)
        self.target_nn = DQN(state_size, action_size).to(self.device)
        self.target_nn.load_state_dict(self.nn.state_dict())

        self.optimizer = optim.Adam(self.nn.parameters(), lr=config['LEARNING_RATE'])
        self.scheduler = lr_scheduler.StepLR(self.optimizer, config['STEP_LR'], config['GAMMA_LR'])

    def update(self, loss, grad_clipping=True):
        self.optimizer.zero_grad()
        loss.backward()
        if self.config['GRAD_CLAMPING']:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.scheduler.step()

    def save(self, folder):
        torch.save(self.nn.state_dict(), folder+'/models/dqn.pth')
        torch.save(self.target_nn.state_dict(), folder+'/models/dqn_target.pth')

    def load(self, folder):
        self.nn.load_state_dict(torch.load(folder+'/models/dqn.pth', map_location='cpu'))
        self.target_nn.load_state_dict(torch.load(folder+'/models/dqn_target.pth', map_location='cpu'))

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            return self.nn(state).cpu().detach().argmax().item()

    def target(self, state):
        state_action = torch.cat([state], -1)
        return self.target_nn(state_action)

    def __call__(self, state):
        state_action = torch.cat([state], -1)
        return self.nn(state_action)


class Model:

    def __init__(self, device, state_size, action_size, folder, config, play=False):

        self.folder = folder
        self.config = config
        self.device = device
        self.memory = NStepsReplayMemory(self.config['MEMORY_SIZE'], self.config['N_STEP'], self.config['GAMMA'])

        self.state_size = state_size
        self.action_size = action_size

        self.agent = Agent(self.state_size, self.action_size, self.device, self.config)

        # Compute gamma^n for n-steps return
        self.gamma_n = self.config["GAMMA"]**self.config['N_STEP']
 
    def select_action(self, state, episode, evaluation=False):
        if evaluation or random.random() > get_epsilon_threshold(episode, self.config):
            return self.agent.select_action(state)
        else:
            return random.randrange(self.action_size)

    def intermediate_reward(self, reward, next_state):
        if self.config['GAME'] == 'Acrobot-v1' and next_state[0] != 0:
            return reward + 1 - next_state[0]

        elif self.config['GAME'] == 'MountainCar-v0' and next_state[0][0] < 0.5:
            return torch.tensor([abs(next_state[0][0]+0.4)], device=self.device)
        elif self.config['GAME'] == 'MountainCar-v0' and next_state[0][0] >= 0.5:
            return torch.tensor([100.0], device=self.device)

        else:
            return torch.tensor([reward], device=self.device)

    def optimize_model(self):
        if len(self.memory) < self.config['BATCH_SIZE']:
            return

        transitions = self.memory.sample(self.config['BATCH_SIZE'])
        batch = list(zip(*transitions))

        states = torch.FloatTensor(batch[0]).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch[3]).to(self.device)
        done = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        current_Q = self.agent.nn(states).gather(1, actions.unsqueeze(1))

        if self.config['DOUBLE_DQN']:
            # ========================== DOUBLE DQN ===============================
            # Compute argmax_a Q(s_{t+1}, a)
            next_actions = self.agent(next_states).argmax(1).unsqueeze(1)

            # Compute Q_target(s_{t+1}, argmax_a Q(s_{t+1}, a)) for a double DQN
            next_Q = self.agent.target(next_states).gather(1, next_actions)

            # Compute the expected Q values : y[i]= r[i] + gamma * Q'(s[i+1], a[i+1])
            target_Q = rewards + done * self.gamma_n * next_Q
            # =====================================================================

        else:
            # ============================== DQN ==================================
            # Compute Q(s_{t+1}) for all next states and select the max 
            next_Q = self.agent.target(next_states).max(1)[0].unsqueeze(1)

            # Compute the expected Q values : y[i]= r[i] + gamma * Q'(s[i+1], a[i+1])
            target_Q = rewards + done * self.gamma_n * next_Q
            # =====================================================================

        # Compute Huber loss
        loss = F.mse_loss(current_Q, target_Q)

        # Optimize the model
        self.agent.update(loss)

        return loss.item()


    def save(self):
        self.agent.save(self.folder)

    def load(self):
        try:
            self.agent.load(self.folder)
        except FileNotFoundError:
            raise Exception("No model has been saved !") from None
