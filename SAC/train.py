import os
import signal
import time
import argparse
try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

import numpy as np
import gym
try:
    import roboschool
except ModuleNotFoundError:
    pass
import matplotlib.pyplot as plt
import yaml

import torch
from tensorboardX import SummaryWriter

from model import Model
from utils import NormalizedActions, get_current_time


print("SAC starting...")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description='Run SAC on ' + config["GAME"])
parser.add_argument('--no_gpu', action='store_false', dest='gpu', help="Don't use GPU")
parser.add_argument('--load', dest='load', type=str, help="Load model")
args = parser.parse_args()


if args.load:
    with open(args.load+'/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

# Create folder and writer to write tensorboard values
folder = f'runs/{config["GAME"].split("-")[0]}_{get_current_time()}'
writer = SummaryWriter(folder)
if not os.path.exists(folder+'/models/'):
    os.mkdir(folder+'/models/')

# Write optional info about the experiment to tensorboard
for k, v in config.items():
    writer.add_text('Config', str(k) + ' : ' + str(v), 0)

# Write a yaml config file in the saving folder
with open(folder+'/config.yaml', 'w') as file:
    yaml.dump(config, file)

# Choose device cpu or cuda if a gpu is available
if args.gpu and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"\033[91m\033[1mDevice : {device.upper()}\nFolder : {folder}\033[0m")
writer.add_text('Device', device, 0)
device = torch.device(device)

# Create gym environment
print("Creating environment...")
env = NormalizedActions(gym.make(config["GAME"]))


def train():

    print("Creating neural networks and optimizers...")
    model = Model(device, folder, config)
    if args.load:
        model.load(args.load)

    # Signal to render evaluation during training by pressing CTRL+Z
    def handler(sig, frame):
        model.evaluate(n_ep=1, render=True)
    signal.signal(signal.SIGTSTP, handler)

    nb_total_steps = 0
    time_beginning = time.time()

    try:
        print("Starting training...")
        nb_episodes = 0
        rewards = []
        eval_rewards = []
        lenghts = []

        for episode in trange(config["MAX_EPISODES"]):

            done = False
            step = 0
            episode_reward = 0

            state = env.reset()

            while not done and step < config["MAX_STEPS"]:

                if nb_total_steps < 10000:
                    action = np.random.uniform(-1, 1, env.action_space.shape[0])

                else:
                    action = model.soft_actor.get_action(state)

                # Perform an action
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                # Save transition into memory
                model.memory.push(state, action, reward, next_state, done)
                state = next_state

                model.optimize()

                step += 1
                nb_total_steps += 1

            rewards.append(episode_reward)
            eval_rewards.append(model.evaluate())
            lenghts.append(step)

            if episode % config["FREQ_PLOT"] == 0:
                writer.add_scalar('rewards', episode_reward, episode)
                writer.add_scalar('eval_rewards', eval_rewards[-1], episode)
                writer.add_scalar('len_episode', lenghts[-1], episode)

                plt.cla()
                plt.title(folder[5:])
                plt.plot(rewards)
                plt.savefig(folder + '/rewards.png')

                plt.cla()
                plt.title(folder[5:])
                plt.plot(eval_rewards)
                plt.savefig(folder + '/eval_rewards.png')

                plt.cla()
                plt.title(folder[5:])
                plt.plot(lenghts)
                plt.savefig(folder + '/lenghts.png')

            nb_episodes += 1

    except KeyboardInterrupt:
        pass

    finally:
        env.close()
        writer.close()
        model.save()
        print("\033[91m\033[1mModel saved in", folder, "\033[0m")

    time_execution = time.time() - time_beginning

    print('---------------------------------------------------\n'
          '---------------------STATS-------------------------\n'
          '---------------------------------------------------\n',
          nb_total_steps, ' steps and updates of the network done\n',
          nb_episodes, ' episodes done\n'
          'Execution time : ', round(time_execution, 2), ' seconds\n'
          '---------------------------------------------------\n'
          'Average nb of steps per second : ', round(nb_total_steps/time_execution, 3), 'steps/s\n'
          'Average duration of one episode : ', round(time_execution/nb_episodes, 3), 's\n'
          '---------------------------------------------------')


if __name__ == '__main__':
    train()
