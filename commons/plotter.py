import itertools

import torch
import numpy as np

import gym
import gym_hypercube
from commons.utils import NormalizedActions

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Plotter:

    def __init__(self, config, device, folder):
        self.device = device
        self.folder = folder
        self.config = config

        self.eval_env = NormalizedActions(gym.make(**config["GAME"]))

        self.nfig = 1
        self.nfig_actor = 1

    def plot_soft_actor_1D(self, soft_actor, pause=False, size=25):
        ss = torch.linspace(-1, 1, size).unsqueeze(1).to(self.device)
        mu, sigma = soft_actor.get_mu_sig(ss)
        mu, sigma = mu.squeeze(), sigma.squeeze()
        ss = ss.cpu().numpy()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(211)
        ax.set_title(f"$tanh(\mu)$")
        ax.plot(ss, np.tanh(mu))
        ax.set_xlabel('State')
        ax.set_ylabel('$tanh(\mu)$')
        ax.set_ylim(-1.05, 1.05)
        ax = fig.add_subplot(212)
        ax.set_title(f"$\sigma$")
        ax.plot(ss, sigma)
        ax.set_xlabel('State')
        ax.set_ylabel('$\sigma$')
        ax.set_ylim(-0.05, 2.05)
        if pause:
            plt.show()
        else:
            plt.savefig(self.folder + f'/Actor{self.nfig_actor:0>3}.jpg')
        plt.close()

        self.nfig_actor += 1

    def plot_actor_1D(self, actor, pause=False, size=25):
        ss = torch.linspace(-1, 1, size).unsqueeze(1).to(self.device)
        a = actor(ss).detach().cpu().numpy()
        ss = ss.cpu().numpy()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_title(f"Action as a function of the state")
        ax.plot(ss, a)
        ax.set_xlabel('State')
        ax.set_ylabel('Action')
        ax.set_ylim(-1.05, 1.05)
        if pause:
            plt.show()
        else:
            plt.savefig(self.folder + f'/Actor{self.nfig_actor:0>3}.jpg')
        plt.close()

        self.nfig_actor += 1

    def plot_Q_1D(self, Qnet, pause=False, size=25):
        if not hasattr(self, 'xx'):
            x, y = np.linspace(-1, 1, size), np.linspace(-1, 1, size)
            self.xx, self.yy = np.meshgrid(x, y)

            self.s = torch.FloatTensor(x).unsqueeze(1).to(self.device)
            self.a = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        Qsa = np.zeros((size, size))
        with torch.no_grad():
            for i in range(size):
                for j in range(size):
                    Qsa[j, i] = Qnet(self.s[i], self.a[j]).detach().cpu().numpy()

        self.in_plot = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.xx, self.yy, Qsa)
        ax.set_title('Q*-value in state-action space')
        ax.set_xlabel('Position')
        ax.set_ylabel('Action')
        ax.set_zlabel('Q')
        # ax.set_zlim(-0.05, 1.05)
        if pause:
            plt.show()
        else:
            plt.savefig(self.folder + f'/Q{"1" if 1 else "2"}_{self.nfig:0>3}.jpg')
        plt.close()
        self.in_plot = False

        self.nfig += 1

    def plot_soft_Q_2D(self, Qnet, soft_actor, pause=False, size=25):
        state = self.eval_env.reset()
        states = [state]
        done = False
        steps = 0
        while not done and steps < self.config['MAX_STEPS']:
            state, r, done, _ = self.eval_env.step(soft_actor.select_action(state))
            states.append(state)
            if pause:
                self.eval_env.render()
            steps += 1
        self.eval_env.close()

        if not hasattr(self, 'xx'):
            x, y = np.linspace(-1, 1, size), np.linspace(-1, 1, size)
            self.xx, self.yy = np.meshgrid(x, y)
            self.s = torch.FloatTensor(list(itertools.product(x, y))).to(self.device)

        with torch.no_grad():
            a, _ = soft_actor(self.s)
            Qsa = Qnet(self.s, a)

        Qsa = Qsa.cpu().numpy().reshape(size, size, order='F')
        a = a.cpu().numpy().reshape(size, size, self.action_size, order='F')

        states = np.array(states)
        with torch.no_grad():
            s = torch.FloatTensor(states).to(self.device)
            aa, _ = soft_actor(s)
            Qsa_states = Qnet(s, aa)
        Qsa_states = Qsa_states.cpu().numpy().squeeze()

        self.in_plot = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(self.xx, self.yy, Qsa)
        ax.quiver(self.xx, self.yy, Qsa, a[:, :, 0], a[:, :, 1], 0, length=0.05, normalize=True, arrow_length_ratio=0.35)
        ax.plot(states[:, 0], states[:, 1], Qsa_states, c='red')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Q(s, $\pi$(s)')
        # ax.set_zlim(0, 1)
        if pause:
            plt.show()
        else:
            plt.savefig(self.folder + f'/Q{self.nfig:0>3}.jpg')
        plt.close()
        self.in_plot = False

        self.nfig += 1
