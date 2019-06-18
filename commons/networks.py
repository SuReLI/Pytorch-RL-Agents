import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal


class DQN(nn.Module):

    def __init__(self, input_size, action_size):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, 16)
        self.hidden2 = nn.Linear(16, 16)
        self.hidden3 = nn.Linear(16, 16)
        self.output = nn.Linear(16, action_size)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        return self.output(x)

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers_size):
        super().__init__()

        self.hiddens = nn.ModuleList([nn.Linear(state_size+action_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], -1)
        for layer in self.hiddens:
            x = torch.relu(layer(x))
        return self.output(x)

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers_size):
        super().__init__()
        self.hiddens = nn.ModuleList([nn.Linear(state_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], action_size)

    def forward(self, x):
        for layer in self.hiddens:
            x = torch.relu(layer(x))
        return torch.tanh(self.output(x))

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_layers_size):
        super().__init__()

        self.hiddens = nn.ModuleList([nn.Linear(state_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], 1)

    def forward(self, x):
        for layer in self.hiddens:
            x = torch.relu(layer(x))
        return self.output(x)

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class SoftActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers_size, device,
                 init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.device = device
        self.action_size = action_size

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.hiddens = nn.ModuleList([nn.Linear(state_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))

        self.mean_output = nn.Linear(hidden_layers_size[-1], action_size)
        self.mean_output.weight.data.uniform_(-init_w, init_w)
        self.mean_output.bias.data.uniform_(-init_w, init_w)

        self.log_std_output = nn.Linear(hidden_layers_size[-1], action_size)
        self.log_std_output.weight.data.uniform_(-init_w, init_w)
        self.log_std_output.bias.data.uniform_(-init_w, init_w)

        self.normal = Normal(0, 1)

    def forward(self, x):
        for layer in self.hiddens:
            x = torch.relu(layer(x))

        mean = self.mean_output(x)
        log_std = torch.tanh(self.log_std_output(x))
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * (log_std+1) / 2

        return mean, log_std

    def evaluate(self, state):
        mean, log_std = self(state)
        std = log_std.exp()

        z = self.normal.sample((self.action_size, )).to(self.device)
        action = torch.tanh(mean + std*z)

        log_prob = Normal(mean, std).log_prob(mean + std*z).sum(dim=1).unsqueeze(1)
        # Cf Annexe C.
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1).unsqueeze(1)

        return action, log_prob

    def get_mu_sig(self, state):
        with torch.no_grad():
            mean, log_std = self(state)
        std = log_std.exp()
        return mean.cpu().numpy(), std.cpu().numpy()

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, log_std = self(state)
            std = log_std.exp()

            z = self.normal.sample((self.action_size, )).to(self.device)
            action = torch.tanh(mean + std*z)

        action = action.cpu().numpy()
        return action[0]

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class Agent:
    def __init__(self, state_size, action_size, device, config):
        self.device = device
        self.config = config

        self.nn = DQN(state_size, action_size).to(self.device)
        self.target_nn = DQN(state_size, action_size).to(self.device)
        self.target_nn.load_state_dict(self.nn.state_dict())
        self.target_nn.eval()

        self.optimizer = optim.Adam(self.nn.parameters(), lr=config['LEARNING_RATE'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config['STEP_LR'], config['GAMMA_LR'])

    def update(self, loss, grad_clipping=True):
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
