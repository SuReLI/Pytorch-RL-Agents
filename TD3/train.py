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
import gym
import yaml

import torch
from tensorboardX import SummaryWriter

from model import Model


print("TD3 starting...")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description='Run TD3 on ' + config["GAME"])
parser.add_argument('--no-gpu', action='store_true', dest='no_gpu', help="Don't use GPU")
args = parser.parse_args()

# Create folder and writer to write tensorboard values
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
expe_name = f'runs/{config["GAME"].split("-")[0]}_{current_time}'
writer = SummaryWriter(expe_name)
if not os.path.exists(expe_name+'/models/'):
    os.mkdir(expe_name+'/models/')

# Write optional info about the experiment to tensorboard
for k, v in config.items():
    writer.add_text('Config', str(k) + ' : ' + str(v), 0)

with open(expe_name+'/config.yaml', 'w') as file:
    yaml.dump(config, file)

# Choose device cpu or cuda if a gpu is available
if not args.no_gpu and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print("\033[91m\033[1mDevice : ", device.upper(), "\033[0m")
writer.add_text('Device', device, 0)
device = torch.device(device)

print("Creating environment...")
env = gym.make(config["GAME"])

LOW_BOUND = int(env.action_space.low[0])
HIGH_BOUND = int(env.action_space.high[0])
STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.shape[0]


def train():

    print("Creating neural networks and optimizers...")
    model = Model(device, STATE_SIZE, ACTION_SIZE, LOW_BOUND, HIGH_BOUND, expe_name, config)

    nb_total_steps = 0
    time_beginning = time.time()

    try:
        print("Starting training...")
        nb_episodes = 0
        rewards = []
        for episode in trange(config["MAX_EPISODES"]):

            done = False
            step = 0
            episode_reward = 0

            state = env.reset()

            while not done and step < config["MAX_STEPS"]:

                if nb_total_steps < 1000:
                    action = env.action_space.sample()

                else:
                    action = model.select_action(state)

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

                actor_loss, critic_loss = model.optimize()

                step += 1
                nb_total_steps += 1

            rewards.append(episode_reward)

            if actor_loss is not None:
                writer.add_scalar('loss/actor_loss', actor_loss, episode)
            if critic_loss is not None:
                writer.add_scalar('loss/critic_loss', critic_loss, episode)
            writer.add_scalar('episode_rewards/actor', episode_reward, episode)

            if nb_episodes % 25 == 0:
                plt.cla()
                plt.plot(rewards)
                plt.savefig(expe_name+'/rewards.png')

            nb_episodes += 1

    except KeyboardInterrupt:
        pass

    finally:
        env.close()
        writer.close()
        model.save()
        print("\033[91m\033[1mModel saved in", expe_name, "\033[0m")

    time_execution = time.time() - time_beginning

    print('---------------------------------------------------\n',
          '---------------------STATS-------------------------\n',
          '---------------------------------------------------\n',
          nb_total_steps, ' steps and updates of the network done\n',
          nb_episodes, ' episodes done\n',
          'Execution time ', round(time_execution, 2), ' seconds\n',
          '---------------------------------------------------\n',
          'Average nb of steps per second : ', round(nb_total_steps/time_execution, 3), 'steps/s\n',
          'Average duration of one episode : ', round(time_execution/nb_episodes, 3), 's\n',
          '---------------------------------------------------')


if __name__ == '__main__':
    train()
