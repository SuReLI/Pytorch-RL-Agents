import sys
import os
import time
from datetime import datetime
import argparse
import gym
import yaml

import torch
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

from model import Model


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description='Run TD3 on ' + config["GAME"])
parser.add_argument('--gpu', help='Use GPU', action='store_true')
parser.add_argument('--eval', help='Model evaluation', action='store_true')
args = parser.parse_args()

# Create folder and writer to write tensorboard values
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
expe_name = f'runs/{current_time}_{config["GAME"][:4]}'
writer = SummaryWriter(expe_name)
if not os.path.exists(expe_name+'/models/'):
    os.mkdir(expe_name+'/models/')

# Write optional info about the experiment to tensorboard
for k, v in config.items():
    writer.add_text('Config', str(k) + ' : ' + str(v), 0)

with open(expe_name+'/config.yaml', 'w') as file:
    yaml.dump(config, file)

# Choose device cpu or cuda if a gpu is available
if args.gpu and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print("\033[91m\033[1mDevice : ", device.upper(), "\033[0m")
device = torch.device(device)

env = gym.make(config["GAME"])

LOW_BOUND = int(env.action_space.low[0])
HIGH_BOUND = int(env.action_space.high[0])

STATE_SIZE = env.observation_space.shape[0]      # state vector size
ACTION_SIZE = env.action_space.shape[0]     # action vector size


def train():

    print("Creating neural networks and optimizers...")
    model = Model(device, STATE_SIZE, ACTION_SIZE, LOW_BOUND, HIGH_BOUND, expe_name, config)

    rewards = []
    nb_total_steps = 0
    time_beginning = time.time()

    try:
        for episode in range(config["MAX_EPISODES"]):

            done = False
            step = 0
            episode_reward = 0

            state = env.reset()

            while not done and step < config["MAX_STEPS"]:

                if nb_total_steps < 1000:
                    action = env.action_space.sample()

                else:
                    action = model.select_action(state)

                    # Add noise
                    noise = np.random.normal(scale=config["EXPLO_SIGMA"], size=ACTION_SIZE)
                    action = np.clip(action+noise, LOW_BOUND, HIGH_BOUND)

                # Perform an action
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                if not done and step == config["MAX_STEPS"] - 1:
                    done = True

                # Save transition into memory
                model.memory.push(state, action, reward, next_state, 1-int(done))
                state = next_state

                step += 1
                nb_total_steps += 1

            for i in range(step):
                model.optimize()

            # writer.add_scalar('episode_rewards/actor', episode_reward, episode)

            print(f"Total T: {nb_total_steps}, "
                  f"Episode Num: {episode}, "
                  f"Episode T: {step}, "
                  f"Reward: {episode_reward}")

            rewards.append(episode_reward)

    except KeyboardInterrupt:
        pass

    finally:
        model.save()

    time_execution = time.time() - time_beginning

    print('---------------------------------------------------\n',
          '---------------------STATS-------------------------\n',
          '---------------------------------------------------\n',
          nb_total_steps, ' steps and updates of the network done\n',
          config["MAX_EPISODES"], ' episodes done\n',
          'Execution time ', round(time_execution, 2), ' seconds\n',
          '---------------------------------------------------\n',
          'Average nb of steps per second : ', round(nb_total_steps/time_execution, 3), 'steps/s\n',
          'Average duration of one episode : ', round(time_execution/config["MAX_EPISODES"], 3), 's\n',
          '---------------------------------------------------')

    plt.figure()
    plt.plot(rewards)

    plt.savefig('results.png')
    # plt.show()


if __name__ == '__main__':
    train()
