import imageio
import gym
try:
    import roboschool
except ModuleNotFoundError:
    pass

import torch
import torch.nn.functional as F

from commons.networks import Actor, Critic
from commons.utils import NormalizedActions, ReplayMemory


class DDPG:

    def __init__(self, device, folder, config):

        self.folder = folder
        self.config = config
        self.device = device
        self.memory = ReplayMemory(self.config["MEMORY_CAPACITY"])
        self.eval_env = NormalizedActions(gym.make(**self.config["GAME"]))

        self.state_size = self.eval_env.observation_space.shape[0]
        self.action_size = self.eval_env.action_space.shape[0]

        self.critic = Critic(self.state_size, self.action_size, device, self.config)
        self.actor = Actor(self.state_size, self.action_size, device, self.config)

    def select_action(self, state, episode=None, evaluation=False):
        assert (episode is not None) or evaluation
        return self.actor.select_action(state)

    def optimize(self):

        if len(self.memory) < self.config["BATCH_SIZE"]:
            return {}

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
        self.critic.update_target(self.config["TAU"])
        self.actor.update_target(self.config["TAU"])

        return {'actor_loss': loss_actor.item(), 'critic_loss': loss_critic.item()}

    def evaluate(self, n_ep=10, render=False, gif=False):
        rewards = []
        if gif:
            writer = imageio.get_writer(self.folder + '/results.gif', duration=0.005)
        try:
            for i in range(n_ep):
                state = self.eval_env.reset()
                reward = 0
                done = False
                steps = 0
                while not done and steps < self.config["MAX_STEPS"]:
                    action = self.select_action(state, evaluation=True)
                    state, r, done, _ = self.eval_env.step(action)
                    if render:
                        self.eval_env.render()
                    if i == 0 and gif:
                        writer.append_data(self.eval_env.render(mode='rgb_array'))
                    reward += r
                    steps += 1
                rewards.append(reward)

        except KeyboardInterrupt:
            pass

        finally:
            self.eval_env.close()
            if gif:
                writer.close()

        score = sum(rewards)/len(rewards) if rewards else 0
        return score

    def save(self):
        print("\033[91m\033[1mModel saved in", self.folder, "\033[0m")
        self.actor.save(self.folder)
        self.critic.save(self.folder)

    def load(self, folder=None):
        if folder is None:
            folder = self.folder
        try:
            self.actor.load(folder)
            self.critic.load(folder)
        except FileNotFoundError:
            raise Exception("No model has been saved !") from None
