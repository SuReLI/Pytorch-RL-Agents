import os
import torch
import torch.optim as optim

from commons.network_modules import QNetwork, CriticNetwork, ActorNetwork


class QAgent:
    def __init__(self, state_size, action_size, device, config):
        self.device = device
        self.config = config

        self.nn = QNetwork(state_size, action_size, config['HIDDEN_LAYERS']).to(self.device)
        self.target_nn = QNetwork(state_size, action_size, config['HIDDEN_LAYERS']).to(self.device)
        self.target_nn.load_state_dict(self.nn.state_dict())
        self.target_nn.eval()

        self.optimizer = optim.Adam(self.nn.parameters(), lr=config['LEARNING_RATE'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config['STEP_LR'], config['GAMMA_LR'])

    def update(self, loss, grad_clipping=False):
        self.optimizer.zero_grad()
        loss.backward()
        if self.config['GRAD_CLAMPING']:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.scheduler.step()

    def update_target(self, tau):
        for target_param, nn_param in zip(self.target_nn.parameters(), self.nn.parameters()):
            target_param.data.copy_((1-tau)*target_param.data + tau*nn_param.data)

    def save(self, folder):
        self.nn.save(os.path.join(folder, 'models/dqn.pth'))
        self.target_nn.save(os.path.join(folder, 'models/dqn_target.pth'))

    def load(self, folder):
        self.nn.load(os.path.join(folder, 'models/dqn.pth'), device=self.device)
        self.target_nn.load(os.path.join(folder, 'models/dqn_target.pth'), device=self.device)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            return self.nn(state).cpu().detach().argmax().item()

    def target(self, state):
        return self.target_nn(state)

    def __call__(self, state):
        return self.nn(state)


class Critic:
    def __init__(self, state_size, action_size, device, config):
        self.device = device

        self.nn = CriticNetwork(state_size, action_size, config['HIDDEN_LAYERS']).to(device)
        self.target_nn = CriticNetwork(state_size, action_size, config['HIDDEN_LAYERS']).to(device)
        self.target_nn.load_state_dict(self.nn.state_dict())
        self.target_nn.eval()

        self.optimizer = optim.Adam(self.nn.parameters(), lr=config["LEARNING_RATE_CRITIC"])

    def update(self, loss, grad_clipping=False):
        self.optimizer.zero_grad()
        loss.backward()
        if grad_clipping:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target(self, tau):
        for target_param, nn_param in zip(self.target_nn.parameters(), self.nn.parameters()):
            target_param.data.copy_((1-tau)*target_param.data + tau*nn_param.data)

    def save(self, folder):
        self.nn.save(os.path.join(folder, 'models/critic.pth'))
        self.target_nn.save(os.path.join(folder, 'models/critic_target.pth'))

    def load(self, folder):
        self.nn.load(os.path.join(folder, 'models/critic.pth'), device=self.device)
        self.target_nn.load(os.path.join(folder, 'models/critic_target.pth'), device=self.device)

    def target(self, state, action):
        return self.target_nn(state, action)

    def __call__(self, state, action):
        return self.nn(state, action)


class Actor:
    def __init__(self, state_size, action_size, device, config):
        self.device = device

        self.nn = ActorNetwork(state_size, action_size, config['HIDDEN_LAYERS']).to(device)
        self.target_nn = ActorNetwork(state_size, action_size, config['HIDDEN_LAYERS']).to(device)
        self.target_nn.load_state_dict(self.nn.state_dict())
        self.target_nn.eval()

        self.optimizer = optim.Adam(self.nn.parameters(), lr=config["LEARNING_RATE_ACTOR"])

    def update(self, loss, grad_clipping=False):
        self.optimizer.zero_grad()
        loss.backward()
        if grad_clipping:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target(self, tau):
        for target_param, nn_param in zip(self.target_nn.parameters(), self.nn.parameters()):
            target_param.data.copy_((1-tau)*target_param.data + tau*nn_param.data)

    def save(self, folder):
        self.nn.save(os.path.join(folder, 'models/actor.pth'))
        self.target_nn.save(os.path.join(folder, 'models/actor_target.pth'))

    def load(self, folder):
        self.nn.load(os.path.join(folder, 'models/actor.pth'), device=self.device)
        self.target_nn.load(os.path.join(folder, 'models/actor_target.pth'), device=self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.nn(state).cpu().detach().numpy()

    def target(self, state):
        return self.target_nn(state)

    def __call__(self, state):
        return self.nn(state)
