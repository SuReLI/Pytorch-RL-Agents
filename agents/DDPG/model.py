import numpy as np
import torch.nn.functional as F

from commons.networks import Actor, Critic
from commons.Abstract_Agent import AbstractAgent


class DDPG(AbstractAgent):

    def __init__(self, device, folder, config):
        super().__init__(device, folder, config)

        self.critic = Critic(self.state_size, self.action_size, device, self.config)
        self.actor = Actor(self.state_size, self.action_size, device, self.config)

    def select_action(self, state, episode=None, evaluation=False):
        assert (episode is not None) or evaluation
        action = self.actor.select_action(state)
        noise = np.random.normal(scale=self.config['EXPLO_SIGMA'], size=self.action_size)
        return np.clip(action+noise, -1, 1)

    def optimize(self):

        if len(self.memory) < self.config['BATCH_SIZE']:
            return {}

        states, actions, rewards, next_states, done = self.get_batch()

        # Compute Q(s,a) using critic network
        current_Q = self.critic(states, actions)

        # Compute deterministic next state action using actor target network
        next_actions = self.actor.target(next_states)

        # Compute next state values at t+1 using target critic network
        target_Q = self.critic.target(next_states, next_actions).detach()

        # Compute expected state action values y[i]= r[i] + Q'(s[i+1], a[i+1])
        target_Q = rewards + (1 - done) * self.config['GAMMA'] * target_Q

        # Critic loss by mean squared error
        loss_critic = F.mse_loss(current_Q, target_Q)

        # Optimize the critic network
        self.critic.update(loss_critic)

        # Optimize actor
        loss_actor = -self.critic(states, self.actor(states)).mean()
        self.actor.update(loss_actor)

        # Soft parameter update
        self.critic.update_target(self.config['TAU'])
        self.actor.update_target(self.config['TAU'])

        return {'actor_loss': loss_actor.item(), 'critic_loss': loss_critic.item()}

    def save(self):
        print("\033[91m\033[1mModel saved in", self.folder, "\033[0m")
        self.actor.save(self.folder)
        self.critic.save(self.folder)

    def load(self, folder=None):
        if folder is None:
            folder = self.folder
        try:
            self.actor.load(folder)
            self.critic.load(folder)
        except FileNotFoundError:
            raise Exception("No model has been saved !") from None
