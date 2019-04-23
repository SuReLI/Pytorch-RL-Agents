import sys
sys.path.extend(["../commons/"])
import random
import math

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from utils import ReplayMemory, update_targets
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

    def __init__(self, device, state_size, action_size, folder, config):

        self.folder = folder
        self.config = config
        self.device = device
        self.memory = ReplayMemory(self.config["MEMORY_CAPACITY"])

        self.state_size = state_size
        self.action_size = action_size

        self.agent = Agent(self.state_size, self.action_size, self.device, self.config)

    def select_action(self, state, episodes):
        threshold = self.config['EPSILON_END'] + (self.config['EPSILON_START'] - self.config['EPSILON_END']) * \
                    math.exp(-1 * episodes / self.config['EPSILON_DECAY'])
        if random.random() > threshold:
            return self.agent.select_action(state)
        else:
            # return random.randrange(self.action_size)
            return torch.tensor([random.randrange(self.action_size)], dtype=torch.long, device=self.device)


    def optimize(self, episode, step):

        if len(self.memory) < self.config["BATCH_SIZE"]:
            return None

        transitions = self.memory.sample(self.config["BATCH_SIZE"])
        # batch = list(zip(*transitions))

        # # Divide memory into different tensors
        # states = torch.FloatTensor(batch[0]).to(self.device)
        # actions = torch.LongTensor(batch[1]).to(self.device)
        # rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        # next_states = torch.FloatTensor(batch[3]).to(self.device)
        # done = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

        # # Compute Q(s,a) using critic network
        # current_Q = self.agent.nn(states).gather(1, actions.unsqueeze(1))

        # # Compute next state values at t+1 using target critic network
        # target_Q = self.agent.target(next_states).detach()
        # target_Q = torch.max(target_Q, 1)[0].unsqueeze(1)

        # # Compute expected state action values y[i]= r[i] + Q'(s[i+1], a[i+1])
        # target_Q = rewards + done * self.config["GAMMA"] * target_Q

        # # Critic loss by mean squared error
        # # loss_critic = F.mse_loss(current_Q, target_Q)
        # loss = F.smooth_l1_loss(current_Q, target_Q)

        # # Optimize the critic network
        # self.agent.update(loss)

        # # Soft parameter update
        # if (episode % self.config["TARGET_UPDATE"] == 0) and (step == 0) :
        #     update_targets(self.agent.target_nn, self.agent.nn, self.config["TAU"])

        # return loss.item()

        from collections import namedtuple
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

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
        state_action_values = self.agent.nn(state_batch).gather(1, action_batch.unsqueeze(1))

        # ============================== DQN ==================================
        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.config['BATCH_SIZE'], device=self.device)
        next_state_values[non_final_mask] = self.agent.target(non_final_next_states).max(1)[0].detach()
        # =====================================================================

        # ========================== DOUBLE DQN ===============================
        # # Compute argmax_a Q(s_{t+1}, a)
        # next_state_values = torch.zeros(self.game_param['BATCH_SIZE'], device=self.device)
        # next_state_actions = self.policy_network(non_final_next_states).max(1)[1].view(len(non_final_next_states), 1).detach()

        # # # Compute Q_target(s_{t+1}, argmax_a Q(s_{t+1}, a))   for a double DQN
        # next_state_values_temp = self.target_network(non_final_next_states).gather(1, next_state_actions).detach()

        # # Compute V(s_{t+1}) for all next states.
        # next_state_values = torch.zeros(self.game_param['BATCH_SIZE'], device=self.device)
        # next_state_values[non_final_mask] = next_state_values_temp.view(1, len(non_final_next_states))
        # =====================================================================

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.config['GAMMA']) + reward_batch
        expected_state_action_values = expected_state_action_values.view(self.config['BATCH_SIZE'], 1)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.agent.optimizer.zero_grad()
        loss.backward()
        if self.config['GRAD_CLAMPING']:
            for param in self.agent.nn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.agent.optimizer.step()
        self.agent.scheduler.step()

        return loss.item()



    def save(self):
        self.agent.save(self.folder)

    def load(self):
        try:
            self.agent.load(self.folder)
        except FileNotFoundError:
            raise Exception("No model has been saved !") from None
