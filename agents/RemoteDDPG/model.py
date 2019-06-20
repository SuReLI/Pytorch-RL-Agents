import sys
sys.path.extend(["../commons/"])

import socket
import threading

import torch
import torch.nn.functional as F

from utils import str_to_list
from networks import Actor, Critic


class Model:

    def __init__(self, device, memory, state_size, action_size, low_bound, high_bound, folder, config):

        self.folder = folder
        self.config = config
        self.device = device
        self.memory = memory

        self.state_size = state_size
        self.action_size = action_size
        self.low_bound = low_bound
        self.high_bound = high_bound

        self.critic = Critic(state_size, action_size, device, self.config)
        self.actor = Actor(state_size, action_size, low_bound, high_bound, device, self.config)

    def select_action(self, state):
        return self.actor.select_action(state)

    def query(self):
        self.thread = threading.Thread(target=self.thread_query)
        self.thread.start()

    def thread_query(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('localhost', 12801))
        sock.listen(5)
        connection, address = sock.accept()

        with connection:
            msg = ''
            while True:
                data = connection.recv(8192)
                if not data:
                    break
                msg += data.decode()

                if '\n' in msg:
                    state, msg = msg.split('\n', 1)
                    send_data = str(list(self.select_action(str_to_list(state))))
                    connection.sendall(send_data.encode())

        sock.close()

    def close(self):
        self.thread.join()

    def optimize(self):

        if len(self.memory) < self.config['BATCH_SIZE']:
            return None, None

        transitions = self.memory.sample(self.config['BATCH_SIZE'])
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
        target_Q = rewards + done*self.config['GAMMA']*target_Q

        # Critic loss by mean squared error
        loss_critic = F.mse_loss(current_Q, target_Q)

        # Optimize the critic network
        self.critic.update(loss_critic)

        # Optimize actor
        loss_actor = -self.critic(states, self.actor(states)).mean()
        self.actor.update(loss_actor)

        # Soft parameter update
        self.critic.update_target(self.config['TAU'])
        self.actor.update_target(self.config['TAU'])

        return loss_actor.item(), loss_critic.item()

    def save(self):
        print("\033[91m\033[1mModel saved in", self.folder, "\033[0m")
        self.actor.save(self.folder)
        self.critic.save(self.folder)

    def load(self):
        try:
            self.actor.load(self.folder)
            self.critic.load(self.folder)
        except FileNotFoundError:
            raise Exception("No model has been saved !") from None
