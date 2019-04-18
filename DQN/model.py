import sys
sys.path.extend(["../commons/"])
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import ReplayMemory, update_targets
from networks import DQN


class Agent:
    def __init__(self, state_size, action_size, device, config):
        self.nn = DQN(self.input_size, self.action_size).to(device)
        self.target_nn = DQN(self.input_size, self.action_size).to(device)
        self.target_nn.load_state_dict(self.nn.state_dict())

        self.optimizer = optim.Adam(self.nn.parameters(), lr=self.game_param['LEARNING_RATE'])
        self.scheduler = lr_scheduler.StepLR(self.optimizer, self.game_param['STEP_LR'], self.game_param['GAMMA_LR'])

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
        # Do a random action ?
        threshold = self.config['EPSILON_END'] + (self.config['EPSILON_START'] - self.config['EPSILON_END']) * \
                    math.exp(-1 * episodes / self.config['EPSILON_DECAY'])

        if random.random() > threshold:
            with torch.no_grad():
                return self.agent.nn(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]],
                                device=self.device, dtype=torch.long)


    def optimize(self):

        if len(self.memory) < self.config["BATCH_SIZE"]:
            return None

        transitions = self.memory.sample(self.config["BATCH_SIZE"])
        batch = list(zip(*transitions))

        # Divide memory into different tensors
        states = torch.FloatTensor(batch[0]).to(self.device)
        actions = torch.FloatTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch[3]).to(self.device)
        done = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

        # Compute Q(s,a) using critic network
        current_Q = self.critic(states, actions)

        # Compute deterministic next state action using actor target network
        next_actions = self.actor.target(next_states)

        # Compute next state values at t+1 using target critic network
        target_Q = self.critic.target(next_states, next_actions).detach()

        # Compute expected state action values y[i]= r[i] + Q'(s[i+1], a[i+1])
        target_Q = rewards + done*self.config["GAMMA"]*target_Q

        # Critic loss by mean squared error
        loss_critic = F.mse_loss(current_Q, target_Q)

        # Optimize the critic network
        self.critic.update(loss_critic)

        # Optimize actor
        loss_actor = -self.critic(states, self.actor(states)).mean()
        self.actor.update(loss_actor)

        # Soft parameter update
        update_targets(self.critic.target_nn, self.critic.nn, self.config["TAU"])
        update_targets(self.actor.target_nn, self.actor.nn, self.config["TAU"])

        return loss_actor.item(), loss_critic.item()



                # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                      device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

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

    def save(self):
        self.actor.save(self.folder)
        self.critic.save(self.folder)

    def load(self):
        try:
            self.actor.load(self.folder)
            self.critic.load(self.folder)
        except FileNotFoundError:
            raise Exception("No model has been saved !") from None
