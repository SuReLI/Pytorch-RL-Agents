import os
import gym
import random
import yaml

import sys
sys.path.extend(["../commons/"])

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from utils import *
from Memory import ReplayMemory
from networks import DQN


class Agent:
    def __init__(self, state_size, action_size, device, config):
        self.device = device

        self.nn = DQN(state_size, action_size).to(self.device)
        self.target_nn = DQN(state_size, action_size).to(self.device)
        self.target_nn.load_state_dict(self.nn.state_dict())

        self.optimizer = optim.Adam(self.nn.parameters(), lr=config['LEARNING_RATE'])
        self.scheduler = lr_scheduler.StepLR(self.optimizer, config['STEP_LR'], config['GAMMA_LR'])

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.nn(state).cpu().detach().max(1)[1]
        # return self.nn(state).cpu().detach().max(0)[1].item()

    def update(self, loss, grad_clipping=True):
        self.optimizer.zero_grad()
        loss.backward()
        if grad_clipping:
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

    def target(self, state):
        state_action = torch.cat([state], -1)
        return self.target_nn(state_action)

    def __call__(self, state):
        state_action = torch.cat([state], -1)
        return self.nn(state_action)


class Model:

    def __init__(self, device, state_size, action_size, folder, config, play=False):
        self.device = device
        self.config = config

        with open('config.yaml', 'r') as stream:
            self.game_param = yaml.load(stream)

        self.steps_done = 0
        self.episodes_done = 0

        self.state_size = state_size
        self.action_size = action_size

        self.policy_network = DQN(self.state_size, self.action_size).to(device)
        self.target_network = DQN(self.state_size, self.action_size).to(device)

        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.policy_network.parameters(),
                                    lr=self.game_param['LEARNING_RATE'])


        self.scheduler = lr_scheduler.StepLR(self.optimizer, self.game_param['STEP_LR'], self.game_param['GAMMA_LR'])
        self.memory = ReplayMemory(10000, self.game_param['N_STEP'], self.game_param['GAMMA'])

        #variables added for step_train function
        self.episodes_reward = []
        self.i_episode_reward = 0

        self.load_network(play=play)
 
    def select_action(self, state, evaluation=False):
        self.steps_done += 1

        if evaluation or random.random() > get_epsilon_threshold(self.episodes_done, self.game_param):
            with torch.no_grad():
                return self.policy_network(state).max(1)[1].view(1, 1)

        else:
            return torch.tensor([[random.randrange(self.action_size)]],
                                device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.game_param['BATCH_SIZE']:
            return

        transitions = self.memory.sample(self.game_param['BATCH_SIZE'])
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                      device=self.device, dtype=torch.uint8)
                                      
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        # ============================== DQN ==================================
        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.game_param['BATCH_SIZE'], device=self.device)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        # =====================================================================

        # ========================== DOUBLE DQN ===============================
        # # Compute argmax_a Q(s_{t+1}, a)
        #next_state_values = torch.zeros(self.game_param['BATCH_SIZE'], device=self.device)
        #next_state_actions = self.policy_network(non_final_next_states).max(1)[1].view(len(non_final_next_states), 1).detach()

        # # Compute Q_target(s_{t+1}, argmax_a Q(s_{t+1}, a))   for a double DQN
        #next_state_values_temp = self.target_network(non_final_next_states).gather(1, next_state_actions).detach()

        # Compute V(s_{t+1}) for all next states.
        #next_state_values = torch.zeros(self.game_param['BATCH_SIZE'], device=self.device)
        #next_state_values[non_final_mask] = next_state_values_temp.view(1, len(non_final_next_states))
        # =====================================================================

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.game_param['GAMMA']**self.game_param['N_STEP']) + reward_batch
        expected_state_action_values = expected_state_action_values.view(self.game_param['BATCH_SIZE'], 1)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        if self.game_param['GRAD_CLAMPING']:
            for param in self.policy_network.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.scheduler.step()

    def intermediate_reward(self, reward, next_state):
        if self.task == 'Acrobot-v1' and next_state[0][0] != 0:
            return torch.tensor([reward + 1 - next_state[0][0]], device=self.device)

        elif self.task == 'MountainCar-v0' and next_state[0][0] < 0.5:
            return torch.tensor([abs(next_state[0][0]+0.4)], device=self.device)
        elif self.task == 'MountainCar-v0' and next_state[0][0] >= 0.5:
            return torch.tensor([100.0], device=self.device)

        else:
            return torch.tensor([reward], device=self.device)


    def play(self, number_run=1):
        for i in range(number_run):
            state = self.env.reset()
            state = torch.tensor([state], dtype=torch.float)
            i_episode_reward = 0
            done = False

            while not done:
                self.env.render()
                action = self.select_action(state, evaluation=True)
                state, reward, done, _ = self.env.step(action.item())
                state = torch.tensor([state], dtype=torch.float)
                i_episode_reward += reward
            print("Episode reward : ", i_episode_reward)


    def save_network(self):
        print("Saving model...")
        if not os.path.exists('../results/DQN/'):
            os.mkdir('../results/DQN/')
        torch.save(self.target_network.state_dict(), '../results/DQN/' + self.task + '_trained_network.pt' )
        print("Model saved !")

    def load_network(self, play=False):
        checkpoint_name = '../results/DQN/' + self.config['GAME'] + '_trained_network.pt'
        if play or ask_loading(checkpoint_name, self.game_param['LOAD']):
            self.policy_network.load_state_dict(torch.load(checkpoint_name))
            print("Network loaded !")
