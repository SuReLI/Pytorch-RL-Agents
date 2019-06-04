
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


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
