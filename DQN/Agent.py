import os
import sys
sys.path.extend(["../commons/"])
import threading
import time
import signal
import gym
import random
import matplotlib.pyplot as plt
import yaml
import csv
import math

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from utils_DQN import *
from utils import ReplayMemory, update_targets
from networks import DQN

from parameters import Parameters


class Agent:
    """Represents an agent acting in an environment by predicting the Q-Value of
    the possible actions with a neural network.

    Args:
        device : a torch device where the computation must be done (CPU or CUDA)
        task : the name of the environment
    """

    def __init__(self, device, task, gui_features=True, play=False):
        self.device = device
        self.task = task

        if self.task == 'Pong-ram-v0':
            self.env = wrap_dqn(gym.make(task))
        else : self.env = gym.make(task)


        with open(task + '.yaml', 'r') as stream:
            self.game_param = yaml.load(stream)

        self.steps_done = 0
        self.episodes_done = 0

        self.input_size = len(self.env.reset())
        self.action_size = self.env.action_space.n

        if self.task == 'Pong-ram-v0':
            self.policy_network = AtariDQN(self.input_size, self.action_size).to(device)
            self.target_network = AtariDQN(self.input_size, self.action_size).to(device)
        else :
            self.policy_network = DQN(self.input_size, self.action_size).to(device)
            self.target_network = DQN(self.input_size, self.action_size).to(device)

        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        if 'NSCartPole' in self.task:
            print('yess')
            self.optimizer = optim.SGD(self.policy_network.parameters(),
                                    lr=self.game_param['LEARNING_RATE'])
        else:
            self.optimizer = optim.Adam(self.policy_network.parameters(),
                                    lr=self.game_param['LEARNING_RATE'])


        self.scheduler = lr_scheduler.StepLR(self.optimizer, self.game_param['STEP_LR'], self.game_param['GAMMA_LR'])
        self.memory = ReplayMemory(10000)

        self.states_csv_path = '../results/DQN/states-'+self.task+'.csv'

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
        # next_state_values = torch.zeros(self.game_param['BATCH_SIZE'], device=self.device)
        # next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        # =====================================================================

        # ========================== DOUBLE DQN ===============================
        # # Compute argmax_a Q(s_{t+1}, a)
        next_state_values = torch.zeros(self.game_param['BATCH_SIZE'], device=self.device)
        next_state_actions = self.policy_network(non_final_next_states).max(1)[1].view(len(non_final_next_states), 1).detach()

        # # Compute Q_target(s_{t+1}, argmax_a Q(s_{t+1}, a))   for a double DQN
        next_state_values_temp = self.target_network(non_final_next_states).gather(1, next_state_actions).detach()

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.game_param['BATCH_SIZE'], device=self.device)
        next_state_values[non_final_mask] = next_state_values_temp.view(1, len(non_final_next_states))
        # =====================================================================

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * Parameters.GAMMA_N) + reward_batch
        expected_state_action_values = expected_state_action_values.view(self.game_param['BATCH_SIZE'], 1)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        if Parameters.GRAD_CLAMPING:
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

    def train(self):
        print("----- Start Training -----")

        if os.path.isfile(self.states_csv_path):
            os.remove(self.states_csv_path)

        self.episodes_done = 0
        stored_states = []
        self.states = []

        i_episode = 1
        self.episodes_done = 1
        
        steps_per_sec = []

        while i_episode < self.game_param['MAX_EPISODES'] :

            time_beginning_ep = time.time()

            # Initialize the environment and state
            state = self.env.reset()
            if self.task == 'NSCartPole-v2':
                state[len(state)-1]=self.game_param['oscillation_magnitude'] * math.sin(state[len(state)-1] * 6.28318530718 / self.game_param['oscillation_period'])

            state = torch.tensor([state], dtype=torch.float, device=self.device)

            i_episode_reward = 0
            done = False
            step = 0

            while step <= self.game_param['MAX_TIMESTEPS'] and not done:

                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action.item())

                if self.task == 'NSCartPole-v2':
                    next_state[len(next_state)-1]=self.game_param['oscillation_magnitude'] * math.sin(next_state[len(next_state)-1] * 6.28318530718 / self.game_param['oscillation_period'])

                # Save the next state in a list, so we can store them in a csv file at the end
                if not done and random.random() < self.game_param['STATE_SAVE_PROBA']:
                    stored_states.append(next_state)

                self.states.append(next_state)
                if len(self.states) > 10000:
                    self.states.pop(0)

                next_state = torch.tensor([next_state], dtype=torch.float, device=self.device)

                reward = self.intermediate_reward(reward, next_state)

                #accumulated reward for each episode
                i_episode_reward += reward.item()

                if done:next_state = None
                # Store the transition in memory
                self.memory.push(state, action, reward, next_state)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()

                step += 1

         
            steps_per_sec.append(round(step/(time.time() - time_beginning_ep),3))
            print(f'Episode {i_episode}, Reward: {i_episode_reward}, '
                      f'Steps: {step}, Epsilon: {get_epsilon_threshold(self.episodes_done, self.game_param):.5}, '
                      f'LR: {self.optimizer.param_groups[0]["lr"]:.4f}, average {round(sum(steps_per_sec[-20:])/20,1) } steps/s')

            # Update the target network
            if i_episode % self.game_param['TARGET_UPDATE'] == 0:
                update_targets(self.target_network, self.policy_network, self.game_param['TAU'])

            i_episode += 1
            self.episodes_done += 1

        print('Complete')

        print("Saving states in ", self.states_csv_path, ' ...')
        #if not os.path.exists(self.states_csv_path):
        #    os.mkdir(self.states_csv_path)
        with open(self.states_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            for s in stored_states :
                writer.writerow(s)

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
        checkpoint_name = '../results/DQN/' + self.task + '_trained_network.pt'
        if play or ask_loading(checkpoint_name):
            self.policy_network.load_state_dict(torch.load(checkpoint_name))
            print("Network loaded !")
