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
        self.optimizer = optim.Adam(self.nn.parameters(), lr=Config.LEARNING_RATE_CRITIC)
        self.target.load_state_dict(self.nn.state_dict())
        self.target.eval()

    def update(self, loss, grad_clipping=True):
        self.optimizer.zero_grad()
        loss.backward()
        if grad_clipping:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def __call__(self, state_action):
        return self.nn(state_action)

    def save()


class Actor:
    def __init__(self, state_size, action_size, low_bound, high_bound, device):
        self.nn = ActorNetwork(state_size, action_size, low_bound, high_bound).to(device)
        self.target = ActorNetwork(state_size, action_size, low_bound, high_bound).to(device)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=Config.LEARNING_RATE_ACTOR)
        self.target.load_state_dict(self.nn.state_dict())
        self.target.eval()

    def update(self, loss, grad_clipping=True):
        self.optimizer.zero_grad()
        loss.backward()
        if grad_clipping:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def __call__(self, state):
        return self.nn(state)


class Model:

    def __init__(self, device, state_size, action_size, low_bound, high_bound):
        self.device = device
        self.memory = ReplayMemory(Config.MEMORY_CAPACITY)

        self.critic = Critic(state_size, action_size, device)
        self.actor = Actor(state_size, action_size, low_bound, high_bound, device)

        self.state_size = state_size
        self.action_size = action_size
        self.low_bound = low_bound
        self.high_bound = high_bound

    def optimize(self, train_critic=True):

        if len(self.memory) < Config.BATCH_SIZE:
            return

        transitions = self.memory.sample(Config.BATCH_SIZE)
        batch = list(zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s_: s_ is not None, batch[3])), device=self.device, dtype=torch.uint8)
        non_final_mask_float = non_final_mask.type(torch.float).view(Config.BATCH_SIZE, 1)

        # Divide memory into different tensors
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1]).view(Config.BATCH_SIZE, -1)
        reward_batch = torch.cat(batch[2]).view(Config.BATCH_SIZE, 1)
        non_final_next_states = torch.cat([s_ for s_ in batch[3] if s_ is not None])

        # Create state-action (s,a) tensor for input into the critic network with taken actions
        state_action = torch.cat([state_batch, action_batch], -1)

        losses = [None]*2
      
        # Compute Q(s,a) using critic network
        state_action_values = self.critic(state_action)
 
        # Compute deterministic next state action using actor target network
        next_action = self.actor.target(non_final_next_states).detach()

        # Compute next timestep state-action (s,a) tensor for non-final next states
        next_state_action = torch.zeros(Config.BATCH_SIZE, self.action_size + self.state_size, device=self.device)
        next_state_action[non_final_mask, :] = torch.cat([non_final_next_states, next_action], -1)

        # Compute next state values at t+1 using target critic network
        next_state_values = self.critic.target(next_state_action).detach()
    
        # Compute expected state action values y[i]= r[i] + Q'(s[i+1], a[i+1])
        expected_state_action_values = reward_batch + non_final_mask_float*Config.GAMMA*next_state_values
        
        # Critic loss by mean squared error
        loss_critic = F.mse_loss(state_action_values, expected_state_action_values)

        # Optimize the critic network
        self.critic.update(loss_critic)

        # Optimize actor
        actor_action = self.actor(state_batch)
        state_actor_action_values = self.critic(torch.cat([state_batch, actor_action], -1))
        loss_actor = -1 * torch.mean(state_actor_action_values)   
        self.actor.update(loss_actor)

        losses[0] = loss_actor.item()
        losses[1] = loss_critic.item()

        # Soft parameter update
        update_targets(self.critic.target, self.critic.nn, Config.TAU)
        update_targets(self.actor.target, self.actor.nn, Config.TAU)

        return losses

    def evaluate(self, n_ep=10, render=False):

        rewards = [0]*n_ep

        for ep in range(n_ep):

            state = self.eval_env.reset()
            state = torch.tensor([state], dtype=torch.float, device=self.device)
            done = False
            step = 0
            while not done and step < Config.MAX_STEPS:

                action = self.actor_star(state).detach()
                state, r, done, _ = self.eval_env.step(action)
                if render:
                    self.eval_env.render()
                state = torch.tensor([state], dtype=torch.float, device=self.device)
                rewards[ep] += r
                step += 1

        return sum(rewards) / n_ep
