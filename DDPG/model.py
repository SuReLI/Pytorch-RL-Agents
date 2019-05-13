import sys
sys.path.extend(["../commons/"])

import gym

import torch
import torch.nn.functional as F

from utils import ReplayMemory, update_targets
from networks import Actor, Critic


class Model:

    def __init__(self, device, state_size, action_size, low_bound, high_bound, folder, config):

        self.folder = folder
        self.config = config
        self.device = device
        self.memory = ReplayMemory(self.config["MEMORY_CAPACITY"])

        self.state_size = state_size
        self.action_size = action_size
        self.low_bound = low_bound
        self.high_bound = high_bound

        self.critic = Critic(state_size, action_size, device, self.config)
        self.actor = Actor(state_size, action_size, low_bound, high_bound, device, self.config)

        self.eval_env = gym.make(self.config["GAME"])

    def select_action(self, state):
        return self.actor.select_action(state)

    def optimize(self):

        if len(self.memory) < self.config["BATCH_SIZE"]:
            return None, None

        transitions = self.memory.sample(self.config["BATCH_SIZE"])
        batch = list(zip(*transitions))

        # Divide memory into different tensors
        states = torch.FloatTensor(batch[0]).to(self.device)
        actions = torch.FloatTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch[3]).to(self.device)
        done = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

        # Compute Q(s,a) using critic network
        current_Q = self.critic(states, actions)

        # Compute deterministic next state action using actor target network
        next_actions = self.actor.target(next_states)

        # Compute next state values at t+1 using target critic network
        target_Q = self.critic.target(next_states, next_actions).detach()

        # Compute expected state action values y[i]= r[i] + Q'(s[i+1], a[i+1])
        target_Q = rewards + done*self.config["GAMMA"]*target_Q

        # Critic loss by mean squared error
        loss_critic = F.mse_loss(current_Q, target_Q)

        # Optimize the critic network
        self.critic.update(loss_critic)

        # Optimize actor
        loss_actor = -self.critic(states, self.actor(states)).mean()
        self.actor.update(loss_actor)

        # Soft parameter update
        update_targets(self.critic.target_nn, self.critic.nn, self.config["TAU"])
        update_targets(self.actor.target_nn, self.actor.nn, self.config["TAU"])

        return loss_actor.item(), loss_critic.item()


    def evaluate(self, n_ep=10, render=False):

        rewards = [0]*n_ep
        steps = [0]*n_ep

        for ep in range(n_ep):

            state = self.eval_env.reset()
            done = False
            step = 0
            while not done and step < self.config["MAX_STEPS"]:

                action = self.select_action(state)
                state, r, done, _ = self.eval_env.step(action)
                if render:
                    self.eval_env.render()
                rewards[ep] += r
                steps[ep] += 1
                step += 1

        return sum(rewards) / n_ep, sum(steps) / n_ep

    def save(self):
        self.actor.save(self.folder)
        self.critic.save(self.folder)

    def load(self):
        try:
            self.actor.load(self.folder)
            self.critic.load(self.folder)
        except FileNotFoundError:
            raise Exception("No model has been saved !") from None
