import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers_size):
        super().__init__()
        self.hiddens = nn.ModuleList([nn.Linear(input_size, hidden_layers_size[0])])
        for i in range(1,len(hidden_layers_size)):
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
        for i in range(1,len(hidden_layers_size)):
            self.hiddens.append(nn.Linear(hidden_layers_size[i-1], hidden_layers_size[i]))
        self.output = nn.Linear(hidden_layers_size[-1], action_size)
        self.low_bound = low_bound
        self.high_bound = high_bound

    def forward(self, x):
        for layer in self.hiddens:
            x = F.relu(layer(x))
        x = self.output(x)
        return (torch.sigmoid(x) * (self.high_bound - self.low_bound)) + self.low_bound


class Critic:
    def __init__(self, state_size, action_size, device, config):
        self.nn = CriticNetwork(state_size + action_size).to(device)
        self.target_nn = CriticNetwork(state_size + action_size).to(device)
        self.target_nn.load_state_dict(self.nn.state_dict())

        self.optimizer = optim.Adam(self.nn.parameters(), lr=config["LEARNING_RATE_CRITIC"])

    def update(self, loss, grad_clipping=True):
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
        self.nn.load_state_dict(torch.load(folder+'/models/critic.pth', map_location='cpu'))
        self.target_nn.load_state_dict(torch.load(folder+'/models/critic_target.pth', map_location='cpu'))

    def target(self, state, action):
        state_action = torch.cat([state, action], -1)
        return self.target_nn(state_action)

    def __call__(self, state, action):
        state_action = torch.cat([state, action], -1)
        return self.nn(state_action)


class Actor:
    def __init__(self, state_size, action_size, low_bound, high_bound, device, config):
        self.device = device

        self.nn = ActorNetwork(state_size, action_size, low_bound, high_bound).to(device)
        self.target_nn = ActorNetwork(state_size, action_size, low_bound, high_bound).to(device)
        self.target_nn.load_state_dict(self.nn.state_dict())

        self.optimizer = optim.Adam(self.nn.parameters(), lr=config["LEARNING_RATE_ACTOR"])

    def update(self, loss, grad_clipping=True):
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
        self.nn.load_state_dict(torch.load(folder+'/models/actor.pth', map_location='cpu'))
        self.target_nn.load_state_dict(torch.load(folder+'/models/actor_target.pth', map_location='cpu'))

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.nn(state).cpu().detach().numpy()

    def target(self, state):
        return self.target_nn(state)

    def __call__(self, state):
        return self.nn(state)
