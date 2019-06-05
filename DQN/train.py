import os
import signal
import time
import argparse
try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

import yaml
import gym
try:
    import roboschool
except ModuleNotFoundError:
    pass
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter

from model import Model
from utils import get_current_time


print("DQN starting...")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description='Run DQN on ' + config["GAME"])
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
env = gym.make(config["GAME"])

STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n


def train():

    print("Creating neural networks and optimizers...")
    model = Model(device, STATE_SIZE, ACTION_SIZE, folder, config)
    if args.load:
        model.load(args.load)

    def handler(sig, frame):
        model.evaluate(n_ep=1, render=True)

    signal.signal(signal.SIGTSTP, handler)

    nb_total_steps = 0
    time_beginning = time.time()

    try:
        print("Starting training...")
        nb_episodes_done = 0
        rewards = []

        for episode in trange(config['MAX_EPISODES']):

            # Initialize the environment and state
            state = env.reset()
            episode_reward = 0
            done = False
            step = 0

            while step <= config['MAX_STEPS'] and not done:

                # Select and perform an action
                action = model.select_action(state, episode)
                next_state, reward, done, _ = env.step(action)
                reward = model.intermediate_reward(reward, next_state)
                episode_reward += reward.item()

                # Store the transition in memory
                model.memory.push(state, action, reward, next_state, 1-int(done))
                state = next_state

                loss = model.optimize()

                step += 1
                nb_total_steps += 1

            rewards.append(episode_reward)

            nb_episodes_done += 1

            # Write scalars to tensorboard
            writer.add_scalar('reward_per_episode', episode_reward, episode)
            writer.add_scalar('steps_per_episode', step, episode)
            if loss is not None:
                writer.add_scalar('loss', loss, episode)

            # Stores .png of the reward graph
            if nb_episodes_done % config["FREQ_PLOT"] == 0:
                plt.cla()
                plt.title(folder[5:])
                plt.plot(rewards)
                plt.savefig(folder+'/rewards.png')

    except KeyboardInterrupt:
        pass

    finally:
        env.close()
        writer.close()
        model.save()
        print("\n\033[91m\033[1mModel saved in", folder, "\033[0m")

    time_execution = time.time() - time_beginning

    print('\n---------------------STATS-------------------------\n',
          nb_total_steps, ' steps and updates of the network done\n',
          nb_episodes_done, ' episodes done\n'
          'Execution time : ', round(time_execution, 2), ' seconds\n'
          'Average nb of steps per second : ', round(nb_total_steps/time_execution, 3), 'steps/s\n'
          'Average duration of one episode : ', round(time_execution/nb_episodes_done, 3), 's\n')


if __name__ == '__main__':
    train()
