import random
import imageio
import gym
try:
    import roboschool
except ModuleNotFoundError:
    pass

import torch
import torch.nn.functional as F

from commons.networks import Agent
from commons.utils import NStepsReplayMemory, get_epsilon_threshold


class DQN:

    def __init__(self, device, folder, config):

        self.folder = folder
        self.config = config
        self.device = device
        self.memory = NStepsReplayMemory(self.config['MEMORY_SIZE'], self.config['N_STEP'], self.config['GAMMA'])
        self.eval_env = gym.make(**self.config['GAME'])

        self.state_size = self.eval_env.observation_space.shape[0]
        self.action_size = self.eval_env.action_space.n

        self.agent = Agent(self.state_size, self.action_size, self.device, self.config)

        # Compute gamma^n for n-steps return
        self.gamma_n = self.config['GAMMA']**self.config['N_STEP']

    def select_action(self, state, episode=None, evaluation=False):
        assert (episode is not None) or evaluation

        if evaluation or random.random() > get_epsilon_threshold(episode, self.config):
            return self.agent.select_action(state)
        else:
            return random.randrange(self.action_size)

    def intermediate_reward(self, reward, next_state):
        if self.config['GAME'] == 'Acrobot-v1' and next_state[0] != 0:
            return reward + 1 - next_state[0]

        elif self.config['GAME'] == 'MountainCar-v0' and next_state[0][0] < 0.5:
            return torch.tensor([abs(next_state[0][0]+0.4)], device=self.device)
        elif self.config['GAME'] == 'MountainCar-v0' and next_state[0][0] >= 0.5:
            return torch.tensor([100.0], device=self.device)

        else:
            return torch.tensor([reward], device=self.device)

    def optimize(self):

        if len(self.memory) < self.config['BATCH_SIZE']:
            return {}

        transitions = self.memory.sample(self.config['BATCH_SIZE'])
        batch = list(zip(*transitions))

        states = torch.FloatTensor(batch[0]).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch[3]).to(self.device)
        done = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        current_Q = self.agent(states).gather(1, actions.unsqueeze(1))

        if self.config['DOUBLE_DQN']:
            # ========================== DOUBLE DQN ===============================
            # Compute argmax_a Q(s_{t+1}, a)
            next_actions = self.agent(next_states).argmax(1).unsqueeze(1)

            # Compute Q_target(s_{t+1}, argmax_a Q(s_{t+1}, a)) for a double DQN
            next_Q = self.agent.target(next_states).gather(1, next_actions)
            # =====================================================================

        else:
            # ============================== DQN ==================================
            # Compute Q(s_{t+1}) for all next states and select the max
            next_Q = self.agent.target(next_states).max(1)[0].unsqueeze(1)
            # =====================================================================

        # Compute the expected Q values : y[i]= r[i] + gamma * Q'(s[i+1], a[i+1])
        target_Q = rewards + (1 - done) * self.gamma_n * next_Q

        loss = F.mse_loss(current_Q, target_Q)

        # Optimize the model
        self.agent.update(loss)

        self.agent.update_target(self.config['TAU'])

        return {'loss': loss.item()}

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
                while not done and steps < self.config['MAX_STEPS']:
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
            if not render:
                raise

        finally:
            self.eval_env.close()
            if gif:
                print(f"Saved gif in {self.folder+'/results.gif'}")
                writer.close()

        score = sum(rewards)/len(rewards) if rewards else 0
        return score

    def save(self):
        print("\033[91m\033[1mModel saved in", self.folder, "\033[0m")
        self.agent.save(self.folder)

    def load(self, folder=None):
        if folder is None:
            folder = self.folder
        try:
            self.agent.load(folder)
        except FileNotFoundError:
            raise Exception("No model has been saved !") from None
