import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


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

    def save(self, folder):
        torch.save(self.nn.state_dict(), folder+'/models/critic.pth')
        torch.save(self.target_nn.state_dict(), folder+'/models/critic_target.pth')

    def load(self, folder):
        self.nn.load_state_dict(torch.load(folder+'/models/critic.pth',
                                           map_location=self.device))
        self.target_nn.load_state_dict(torch.load(folder+'/models/critic_target.pth',
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

    def save(self, folder):
        torch.save(self.nn.state_dict(), folder+'/models/actor.pth')
        torch.save(self.target_nn.state_dict(), folder+'/models/actor_target.pth')

    def load(self, folder):
        self.nn.load_state_dict(torch.load(folder+'/models/actor.pth',
                                           map_location=self.device))
        self.target_nn.load_state_dict(torch.load(folder+'/models/actor_target.pth',
                                                  map_location=self.device))

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.nn(state).cpu().detach().numpy()

    def target(self, state):
        return self.target_nn(state)

    def __call__(self, state):
        return self.nn(state)
