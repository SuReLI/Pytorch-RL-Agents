import sys
import os
import time
from datetime import datetime
import argparse
try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

import numpy as np
import matplotlib.pyplot as plt
# import roboschool
import gym
import yaml

import torch
from tensorboardX import SummaryWriter

from model import Model

print("DDPG starting...")

with open('config.yaml', 'r') as stream:
    config = yaml.load(stream)

parser = argparse.ArgumentParser(description='Run DDPG on ' + config['GAME'])
parser.add_argument('--gpu', action='store_true', help='Use GPU')
args = parser.parse_args()

# Create folder and writer to write tensorboard values
if not os.path.exists('runs'):
    os.mkdir('runs')
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
expe_name = 'runs/'+current_time+'_'+config['GAME'][:4]
writer = SummaryWriter(expe_name)
if not os.path.exists(expe_name+'/models/'):
    os.mkdir(expe_name+'/models/')

# Write optional info about the experiment to tensorboard
for k, v in config.items():
    writer.add_text('Config', str(k) + ' : ' + str(v), 0)

with open(expe_name+'/config.yaml', 'w') as stream:
    yaml.dump(config, stream)

# Choose device cpu or cuda if a gpu is available
if args.gpu and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("\033[91m\033[1mDevice : ", device.upper(), "\033[0m")
writer.add_text('Device', device, 0)
device = torch.device(device)


# Create gym environment
print("Creating environment...")
env = gym.make(config['GAME'])

LOW_BOUND = int(env.action_space.low[0])
HIGH_BOUND = int(env.action_space.high[0])
STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.shape[0]

print("Creating neural networks and optimizers...")
model = Model(device, STATE_SIZE, ACTION_SIZE, LOW_BOUND, HIGH_BOUND, expe_name)

losses_names = ['actor', 'critic']
episode_reward = []
nb_total_steps = 0
time_beginning = time.time()

try:

    print("Starting training...")
    eval_reward = []
    nb_episodes_done = 0
    for i_episode in trange(config['MAX_EPISODES']):

        state = env.reset()
        state = torch.tensor([state], dtype=torch.float, device=device)
        done = False
        step = 0
        current_reward = 0

        while not done and step < config['MAX_STEPS']:

            action = model.actor(state).detach()

            noise = np.random.normal(scale=config['EPSILON'], size=ACTION_SIZE)
            action += torch.tensor([noise], dtype=torch.float, device=device)
            action = torch.clamp(action, LOW_BOUND, HIGH_BOUND) 

            # Perform an action
            next_state, reward, done, _ = env.step(action.numpy()[0])
            next_state = torch.tensor([next_state], dtype=torch.float, device=device)
            if done:
                next_state = None

            current_reward += reward

            reward = torch.tensor([reward], dtype=torch.float, device=device)

            # Save transition into memory
            model.memory.push(state, action, reward, next_state)
            state = next_state

            optimizer_losses = model.optimize(writer)
            
            step += 1
            nb_total_steps += 1

        writer.add_scalar('episode_rewards/actor', current_reward, i_episode)
        if optimizer_losses :
                for i in range(len(optimizer_losses)):
                    if optimizer_losses[i] is not None:
                        writer.add_scalar('loss/'+losses_names[i], optimizer_losses[i], i_episode)
        nb_episodes_done += 1

except KeyboardInterrupt:
    pass

model.save()
print("\033[91m\033[1mModel saved in", expe_name, "\033[0m")

time_execution = time.time() - time_beginning

print('---------------------------------------------------\n'
      '---------------------STATS-------------------------\n'
      '---------------------------------------------------\n',
      nb_total_steps, ' steps and updates of the network done\n',
      nb_episodes_done, ' episodes done\n'
      'Execution time ', round(time_execution, 2), ' seconds\n'
      '---------------------------------------------------\n'
      'Average nb of steps per second : ', round(nb_total_steps/time_execution, 3), 'steps/s\n'
      'Average duration of one episode : ', round(time_execution/nb_episodes_done, 3), 's\n'
      '---------------------------------------------------')

writer.close()
