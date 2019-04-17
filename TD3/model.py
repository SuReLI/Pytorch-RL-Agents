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
        self.target = CriticNetwork(state_size + action_size).to(device)
        self.target.load_state_dict(self.nn.state_dict())
        # self.target.eval()
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
        torch.save(self.target.state_dict(), 'models/critic_target.pth')

    def load(self):
        self.nn.load_state_dict(torch.load('models/critic.pth', map_location='cpu'))
        self.target.load_state_dict(torch.load('models/critic_target.pth', map_location='cpu'))

    def __call__(self, state_action):
        return self.nn(state_action)


class Actor:
    def __init__(self, state_size, action_size, low_bound, high_bound, device):
        self.nn = ActorNetwork(state_size, action_size, low_bound, high_bound).to(device)
        self.target = ActorNetwork(state_size, action_size, low_bound, high_bound).to(device)
        self.target.load_state_dict(self.nn.state_dict())
        # self.target.eval()
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
        torch.save(self.target.state_dict(), 'models/actor_target.pth')

    def load(self):
        self.nn.load_state_dict(torch.load('models/actor.pth', map_location='cpu'))
        self.target.load_state_dict(torch.load('models/actor_target.pth', map_location='cpu'))

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

    def optimize(self):

        if len(self.memory) < Config.BATCH_SIZE:
            return

        transitions = self.memory.sample(Config.BATCH_SIZE)
        batch = list(zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # done_mask = torch.tensor(tuple(map(lambda s_: s_ is not None, batch[3])), device=self.device, dtype=torch.uint8)
        # done_mask_f = done_mask.type(torch.float).view(Config.BATCH_SIZE, 1)

        # Divide memory into different tensors
        state = torch.cat(batch[0]).to(self.device)
        action = torch.cat(batch[1]).view(Config.BATCH_SIZE, -1).to(self.device)
        reward = torch.cat(batch[2]).view(Config.BATCH_SIZE, 1).to(self.device)
        next_states = torch.cat(batch[3]).to(self.device)
        done = torch.tensor(batch[4], dtype=torch.float).view(Config.BATCH_SIZE, 1).to(self.device)
        # nf_next_states = torch.cat([s_ for s_ in batch[3] if s_ is not None])

        # Create state-action (s,a) tensor for input into the critic network with taken actions
        state_action = torch.cat([state, action], -1)

        # Compute Q(s,a) using critic network
        current_Qa = self.critic_A(state_action)
        current_Qb = self.critic_B(state_action)

        # Compute deterministic next state action using actor target network
        next_action = self.actor.target(next_states).to(self.device).detach()
        noise = torch.normal(torch.zeros(Config.BATCH_SIZE), Config.UPDATE_SIGMA*torch.ones(Config.BATCH_SIZE))
        noise = noise.clamp(-Config.UPDATE_CLIP, Config.UPDATE_CLIP).view(Config.BATCH_SIZE, 1).to(self.device)
        next_action = torch.clamp(next_action+noise, self.low_bound, self.high_bound)

        # Compute next timestep state-action (s,a) tensor for non-final next states
        # next_state_action = torch.zeros(Config.BATCH_SIZE, self.action_size+self.state_size, device=self.device)
        # next_state_action[done_mask, :] = torch.cat([nf_next_states, next_action], -1)
        next_state_action = torch.cat([next_states, next_action], 1)

        # Compute next state values at t+1 using target critic network
        target_Qa = self.critic_A.target(next_state_action).detach()
        target_Qb = self.critic_B.target(next_state_action).detach()
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
            actor_action = self.actor(state)
            state_actor_action_values = self.critic_A(torch.cat([state, actor_action], -1))
            loss_actor = -torch.mean(state_actor_action_values)

            self.actor.update(loss_actor, grad_clipping=False)

            update_targets(self.actor.target, self.actor.nn, Config.TAU)
            
            update_targets(self.critic_A.target, self.critic_A.nn, Config.TAU)
            update_targets(self.critic_B.target, self.critic_B.nn, Config.TAU)

        self.update_step += 1

    def evaluate(self, n_ep=10, render=False):

        rewards = [0]*n_ep

        for ep in range(n_ep):

            state = self.eval_env.reset()
            state = torch.tensor([state], dtype=torch.float, device=self.device)
            done = False
            step = 0
            while not done and step < Config.MAX_STEPS:

                action = self.actor(state).detach()
                state, r, done, _ = self.eval_env.step(action.numpy()[0])
                if render:
                    self.eval_env.render()
                state = torch.tensor([state], dtype=torch.float, device=self.device)
                rewards[ep] += r
                step += 1

        return sum(rewards) / n_ep

    def save(self):
        self.actor.save()
        self.critic_A.save()

    def load(self):
        self.actor.load()
        self.critic_A.load()
