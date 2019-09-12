import numpy as np

import torch
import torch.nn.functional as F

from commons.networks import Actor, Critic
from commons.Abstract_Agent import AbstractAgent


class TD3(AbstractAgent):

    def __init__(self, device, folder, config):
        super().__init__(device, folder, config)

        self.critic_A = Critic(self.state_size, self.action_size, device, config)
        self.critic_B = Critic(self.state_size, self.action_size, device, config)
        self.actor = Actor(self.state_size, self.action_size, device, config)

        self.update_step = 0

    def select_action(self, state, episode=None, evaluation=False):
        assert (episode is not None) or evaluation
        action = self.actor.select_action(state)
        noise = np.random.normal(scale=self.config['EXPLO_SIGMA'], size=self.action_size)
        return np.clip(action+noise, -1, 1)

    def optimize(self):

        if len(self.memory) < self.config['BATCH_SIZE']:
            return {}

        self.update_step += 1
        states, actions, rewards, next_states, done = self.get_batch()

        # Compute Q(s,a) using critic network
        current_Qa = self.critic_A(states, actions)
        current_Qb = self.critic_B(states, actions)

        # Compute deterministic next state action using actor target network
        next_actions = self.actor.target(next_states)
        noise = torch.normal(0, self.config['UPDATE_SIGMA']*torch.ones([self.config['BATCH_SIZE'], 1]))
        noise = noise.clamp(-self.config['UPDATE_CLIP'], self.config['UPDATE_CLIP']).to(self.device)
        next_actions = torch.clamp(next_actions+noise, -1, 1)

        # Compute next state values at t+1 using target critic network
        target_Qa = self.critic_A.target(next_states, next_actions).detach()
        target_Qb = self.critic_B.target(next_states, next_actions).detach()
        target_Q = torch.min(target_Qa, target_Qb)

        # Compute expected state action values y[i]= r[i] + Q'(s[i+1], a[i+1])
        target_Q = rewards + (1 - done) * self.config['GAMMA'] * target_Q

        # loss_critic = F.mse_loss(current_Qa, target_Q) + F.mse_loss(current_Qb, target_Q)
        loss_critic_A = F.mse_loss(current_Qa, target_Q)
        loss_critic_B = F.mse_loss(current_Qb, target_Q)

        self.critic_A.update(loss_critic_A)
        self.critic_B.update(loss_critic_B)

        # Optimize actor every 2 steps
        if self.update_step % 2 == 0:
            loss_actor = -self.critic_A(states, self.actor(states)).mean()

            self.actor.update(loss_actor)

            self.actor.update_target(self.config['TAU'])

            self.critic_A.update_target(self.config['TAU'])
            self.critic_B.update_target(self.config['TAU'])

            return {'Q1_loss': loss_critic_A.item(), 'Q2_loss': loss_critic_B.item(),
                    'actor_loss': loss_actor.item()}

        else:
            return {'Q1_loss': loss_critic_A.item(), 'Q2_loss': loss_critic_B.item()}

    def save(self):
        print("\033[91m\033[1mModel saved in", self.folder, "\033[0m")
        self.actor.save(self.folder)
        self.critic_A.save(self.folder)

    def load(self, folder=None):
        if folder is None:
            folder = self.folder
        try:
            self.actor.load(folder)
            self.critic_A.load(folder)
        except FileNotFoundError:
            raise Exception("No model has been saved !") from None
