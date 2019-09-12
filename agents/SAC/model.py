import numpy as np

import torch

from commons.network_modules import ValueNetwork, CriticNetwork, SoftActorNetwork
from commons.plotter import Plotter
from commons.Abstract_Agent import AbstractAgent


class SAC(AbstractAgent):

    def __init__(self, device, folder, config):
        super().__init__(device, folder, config)

        self.value_net = ValueNetwork(self.state_size, self.config['HIDDEN_VALUE_LAYERS']).to(device)
        self.target_value_net = ValueNetwork(self.state_size, self.config['HIDDEN_VALUE_LAYERS']).to(device)
        self.soft_Q_net1 = CriticNetwork(self.state_size, self.action_size, self.config['HIDDEN_Q_LAYERS']).to(device)
        self.soft_Q_net2 = CriticNetwork(self.state_size, self.action_size, self.config['HIDDEN_Q_LAYERS']).to(device)
        self.soft_actor = SoftActorNetwork(self.state_size, self.action_size, self.config['HIDDEN_PI_LAYERS'], device).to(device)
        self.target_value_net.eval()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.config['VALUE_LR'])
        self.soft_q_optimizer1 = torch.optim.Adam(self.soft_Q_net1.parameters(), lr=self.config['SOFTQ_LR'])
        self.soft_q_optimizer2 = torch.optim.Adam(self.soft_Q_net2.parameters(), lr=self.config['SOFTQ_LR'])
        self.soft_actor_optimizer = torch.optim.Adam(self.soft_actor.parameters(), lr=self.config['ACTOR_LR'])

        self.q_criterion1 = torch.nn.MSELoss()
        self.q_criterion2 = torch.nn.MSELoss()
        self.value_criterion = torch.nn.MSELoss()

        if self.config['AUTO_ALPHA']:
            self.target_entropy = -np.prod(self.eval_env.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config['ALPHA_LR'])

        self.plotter = Plotter(config, device, folder)

    def select_action(self, state, episode=None, evaluation=False):
        assert (episode is not None) or evaluation
        return self.soft_actor.select_action(state)

    def optimize(self):

        if len(self.memory) < self.config['BATCH_SIZE']:
            return {}

        states, actions, rewards, next_states, done = self.get_batch()

        current_Q1 = self.soft_Q_net1(states, actions)
        current_Q2 = self.soft_Q_net2(states, actions)
        current_V = self.value_net(states)
        new_actions, log_prob = self.soft_actor.evaluate(states)

        # Compute the next value of alpha
        if self.config['AUTO_ALPHA']:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 0.2

        next_V = self.target_value_net(next_states)
        target_Q = rewards + (1 - done) * self.config['GAMMA'] * next_V

        expected_new_Q1 = self.soft_Q_net1(states, new_actions)
        expected_new_Q2 = self.soft_Q_net2(states, new_actions)
        expected_new_Q = torch.min(expected_new_Q1, expected_new_Q2)
        target_V = expected_new_Q - alpha * log_prob

        loss_Q1 = self.q_criterion1(current_Q1, target_Q.detach())
        loss_Q2 = self.q_criterion2(current_Q2, target_Q.detach())
        loss_V = self.value_criterion(current_V, target_V.detach())
        loss_actor = (alpha * log_prob - expected_new_Q1).mean()

        self.soft_q_optimizer1.zero_grad()
        loss_Q1.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        loss_Q2.backward()
        self.soft_q_optimizer2.step()

        self.value_optimizer.zero_grad()
        loss_V.backward()
        self.value_optimizer.step()

        self.soft_actor_optimizer.zero_grad()
        loss_actor.backward()
        self.soft_actor_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data*(1.0-self.config['TAU']) + param.data*self.config['TAU'])

        return {'Q1_loss': loss_Q1.item(), 'Q2_loss': loss_Q2.item(),
                'V_loss': loss_V.item(), 'actor_loss': loss_actor.item()}

    def save(self):
        print("\033[91m\033[1mModel saved in", self.folder, "\033[0m")
        self.value_net.save(self.folder + '/models/value.pth')
        self.target_value_net.save(self.folder + '/models/value_target.pth')
        self.soft_Q_net1.save(self.folder + '/models/soft_Q.pth')
        self.soft_actor.save(self.folder + '/models/soft_actor.pth')

    def load(self):
        try:
            self.value_net.load(self.folder + '/models/value.pth', self.device)
            self.target_value_net.load(self.folder + '/models/value_target.pth', self.device)
            self.soft_Q_net1.load(self.folder + '/models/soft_Q.pth', self.device)
            self.soft_Q_net2.load(self.folder + '/models/soft_Q.pth', self.device)
            self.soft_actor.load(self.folder + '/models/soft_actor.pth', self.device)
        except FileNotFoundError:
            raise Exception("No model has been saved !") from None

    def plot_Q(self, pause=False):
        if self.state_size == 1 and self.action_size == 1:
            self.plotter.plot_soft_actor_1D(self.soft_actor, pause)
            self.plotter.plot_Q_1D(self.soft_Q_net1, pause)

        if self.state_size == 2 and self.action_size == 2:
            self.plotter.plot_soft_Q_2D(self.soft_Q_net1, self.soft_actor, pause)
