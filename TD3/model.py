import torch
import torch.optim as optim
import torch.nn.functional as F

import gym

from config import Config
from utils import ReplayMemory, update_targets
from networks import ActorNetwork, CriticNetwork


class Critic:
    def __init__(self, state_size, action_size, device):
        self.nn = CriticNetwork(state_size + action_size).to(device)
        self.target_nn = CriticNetwork(state_size + action_size).to(device)
        self.target_nn.load_state_dict(self.nn.state_dict())

        self.optimizer = optim.Adam(self.nn.parameters(), lr=Config.LEARNING_RATE_CRITIC)

    def update(self, loss, grad_clipping=True):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if grad_clipping:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self):
        torch.save(self.nn.state_dict(), 'models/critic.pth')
        torch.save(self.target_nn.state_dict(), 'models/critic_target.pth')

    def load(self):
        self.nn.load_state_dict(torch.load('models/critic.pth', map_location='cpu'))
        self.target_nn.load_state_dict(torch.load('models/critic_target.pth', map_location='cpu'))

    def target(self, state, action):
        state_action = torch.cat([state, action], -1)
        return self.target_nn(state_action)

    def __call__(self, state, action):
        state_action = torch.cat([state, action], -1)
        return self.nn(state_action)


class Actor:
    def __init__(self, state_size, action_size, low_bound, high_bound, device):
        self.device = device

        self.nn = ActorNetwork(state_size, action_size, low_bound, high_bound).to(device)
        self.target_nn = ActorNetwork(state_size, action_size, low_bound, high_bound).to(device)
        self.target_nn.load_state_dict(self.nn.state_dict())

        self.optimizer = optim.Adam(self.nn.parameters(), lr=Config.LEARNING_RATE_ACTOR)

    def update(self, loss, grad_clipping=True):
        self.optimizer.zero_grad()
        loss.backward()
        if grad_clipping:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self):
        torch.save(self.nn.state_dict(), 'models/actor.pth')
        torch.save(self.target_nn.state_dict(), 'models/actor_target.pth')

    def load(self):
        self.nn.load_state_dict(torch.load('models/actor.pth', map_location='cpu'))
        self.target_nn.load_state_dict(torch.load('models/actor_target.pth', map_location='cpu'))

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.nn(state).cpu().detach().numpy()

    def target(self, state):
        return self.target_nn(state)

    def __call__(self, state):
        return self.nn(state)


class Model:

    def __init__(self, device, state_size, action_size, low_bound, high_bound):
        self.device = device
        self.memory = ReplayMemory(Config.MEMORY_CAPACITY)

        self.state_size = state_size
        self.action_size = action_size
        self.low_bound = low_bound
        self.high_bound = high_bound

        self.critic_A = Critic(state_size, action_size, device)
        self.critic_B = Critic(state_size, action_size, device)
        self.actor = Actor(state_size, action_size, low_bound, high_bound, device)

        self.eval_env = gym.make(Config.GAME)

        self.update_step = 0

    def select_action(self, state):
        return self.actor.select_action(state)

    def optimize(self):

        if len(self.memory) < Config.BATCH_SIZE:
            return

        transitions = self.memory.sample(Config.BATCH_SIZE)
        batch = list(zip(*transitions))

        # Divide memory into different tensors
        state = torch.FloatTensor(batch[0]).to(self.device)
        action = torch.FloatTensor(batch[1]).to(self.device)
        reward = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch[3]).to(self.device)
        done = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

        # Compute Q(s,a) using critic network
        current_Qa = self.critic_A(state, action)
        current_Qb = self.critic_B(state, action)

        # Compute deterministic next state action using actor target network
        next_action = self.actor.target(next_states)
        noise = torch.normal(0, Config.UPDATE_SIGMA*torch.ones([Config.BATCH_SIZE, 1]))
        noise = noise.clamp(-Config.UPDATE_CLIP, Config.UPDATE_CLIP).to(self.device)
        next_action = torch.clamp(next_action+noise, self.low_bound, self.high_bound)

        # Compute next state values at t+1 using target critic network
        target_Qa = self.critic_A.target(next_states, next_action).detach()
        target_Qb = self.critic_B.target(next_states, next_action).detach()
        target_Q = torch.min(target_Qa, target_Qb)

        # Compute expected state action values y[i]= r[i] + Q'(s[i+1], a[i+1])
        target_Q = reward + done*Config.GAMMA*target_Q

        # loss_critic = F.mse_loss(current_Qa, target_Q) + F.mse_loss(current_Qb, target_Q)
        loss_critic_A = F.mse_loss(current_Qa, target_Q)
        loss_critic_B = F.mse_loss(current_Qb, target_Q)

        self.critic_A.update(loss_critic_A, grad_clipping=False)
        self.critic_B.update(loss_critic_B, grad_clipping=False)

        # Optimize actor every 2 steps
        if self.update_step % 2 == 0:
            loss_actor = -self.critic_A(state, self.actor(state)).mean()

            self.actor.update(loss_actor, grad_clipping=False)

            update_targets(self.actor.target_nn, self.actor.nn, Config.TAU)

            update_targets(self.critic_A.target_nn, self.critic_A.nn, Config.TAU)
            update_targets(self.critic_B.target_nn, self.critic_B.nn, Config.TAU)

        self.update_step += 1

    def evaluate(self, n_ep=10, render=False):

        rewards = [0]*n_ep

        for ep in range(n_ep):

            state = self.eval_env.reset()
            done = False
            step = 0
            while not done and step < Config.MAX_STEPS:

                action = self.actor(state)
                state, r, done, _ = self.eval_env.step(action)
                if render:
                    self.eval_env.render()
                rewards[ep] += r
                step += 1

        return sum(rewards) / n_ep

    def save(self):
        self.actor.save()
        self.critic_A.save()

    def load(self):
        self.actor.load()
        self.critic_A.load()
