import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class DQN(nn.Module):

    def __init__(self, input_size, action_size):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, 16)
        self.hidden2 = nn.Linear(16, 16)
        self.hidden3 = nn.Linear(16, 16)
        self.output = nn.Linear(16, action_size)

    def forward(self, x):

        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        return self.output(x)


class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers_size):
        super().__init__()
        self.hiddens = nn.ModuleList([nn.Linear(input_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], 1)

    def forward(self, x):
        for layer in self.hiddens:
            x = F.relu(layer(x))
        return self.output(x)


class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, low_bound, high_bound, hidden_layers_size):
        super().__init__()
        self.hiddens = nn.ModuleList([nn.Linear(state_size, hidden_layers_size[0])])
        for i in range(1, len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], action_size)
        self.low_bound = low_bound
        self.high_bound = high_bound

    def forward(self, x):
        for layer in self.hiddens:
            x = F.relu(layer(x))
        x = (torch.tanh(self.output(x)) + 1) / 2
        return (x * (self.high_bound - self.low_bound)) + self.low_bound


class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, init_w=3e-3):
        super().__init__()

        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class SoftQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, init_w=3e-3):
        super().__init__()

        self.linear1 = nn.Linear(state_size + action_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, device,
                 init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.device = device
        self.action_size = action_size

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, action_size)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, action_size)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * (log_std+1) / 2
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample((self.action_size, )).to(self.device)
        action = torch.tanh(mean + std*z)

        log_prob = Normal(mean, std).log_prob(mean + std*z).sum(dim=1).unsqueeze(1)
        # Cf Annexe C.
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1).unsqueeze(1)

        return action, log_prob

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample((self.action_size, )).to(self.device)
        action = torch.tanh(mean + std*z)

        action = action.detach().cpu().numpy()
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
        torch.save(self.nn.state_dict(), os.path.join(folder, 'models/dqn.pth'))
        torch.save(self.target_nn.state_dict(), os.path.join(folder, 'models/dqn_target.pth'))

    def load(self, folder):
        self.nn.load_state_dict(torch.load(os.path.join(folder, 'models/dqn.pth'), map_location='cpu'))
        self.target_nn.load_state_dict(torch.load(os.path.join(folder, 'models/dqn_target.pth'), map_location='cpu'))

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

        self.nn = CriticNetwork(state_size + action_size, config['HIDDEN_LAYERS']).to(device)
        self.target_nn = CriticNetwork(state_size + action_size, config['HIDDEN_LAYERS']).to(device)
        self.target_nn.load_state_dict(self.nn.state_dict())

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
        torch.save(self.nn.state_dict(), os.path.join(folder, 'models/critic.pth'))
        torch.save(self.target_nn.state_dict(), os.path.join(folder, 'models/critic_target.pth'))

    def load(self, folder):
        self.nn.load_state_dict(torch.load(os.path.join(folder, 'models/critic.pth'),
                                           map_location=self.device))
        self.target_nn.load_state_dict(torch.load(os.path.join(folder, 'models/critic_target.pth'),
                                                  map_location=self.device))

    def target(self, state, action):
        state_action = torch.cat([state, action], -1)
        return self.target_nn(state_action)

    def __call__(self, state, action):
        state_action = torch.cat([state, action], -1)
        return self.nn(state_action)


class Actor:
    def __init__(self, state_size, action_size, low_bound, high_bound, device, config):
        self.device = device

        self.nn = ActorNetwork(state_size, action_size, low_bound, high_bound, config['HIDDEN_LAYERS']).to(device)
        self.target_nn = ActorNetwork(state_size, action_size, low_bound, high_bound, config['HIDDEN_LAYERS']).to(device)
        self.target_nn.load_state_dict(self.nn.state_dict())

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
        torch.save(self.nn.state_dict(), os.path.join(folder, 'models/actor.pth'))
        torch.save(self.target_nn.state_dict(), os.path.join(folder, 'models/actor_target.pth'))

    def load(self, folder):
        self.nn.load_state_dict(torch.load(os.path.join(folder, 'models/actor.pth'),
                                           map_location=self.device))
        self.target_nn.load_state_dict(torch.load(os.path.join(folder, 'models/actor_target.pth'),
                                                  map_location=self.device))

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.nn(state).cpu().detach().numpy()

    def target(self, state):
        return self.target_nn(state)

    def __call__(self, state):
        return self.nn(state)
