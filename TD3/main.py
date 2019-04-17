import sys
import time
import gym

import torch

import numpy as np
import matplotlib.pyplot as plt

import argparse

from model import Model
from config import Config


parser = argparse.ArgumentParser(description='Run TD3 on ' + Config.GAME)
parser.add_argument('--gpu', help='Use GPU', action='store_true')
parser.add_argument('--eval', help='Model evaluation', action='store_true')
args = parser.parse_args()

# Choose device cpu or cuda if a gpu is available
if args.gpu and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print("\033[91m\033[1mDevice : ", device.upper(), "\033[0m")
device = torch.device(device)

env = gym.make(Config.GAME)

LOW_BOUND = int(env.action_space.low[0])
HIGH_BOUND = int(env.action_space.high[0])

STATE_SIZE = env.observation_space.shape[0]      # state vector size
ACTION_SIZE = env.action_space.shape[0]     # action vector size


model = Model(device, STATE_SIZE, ACTION_SIZE, LOW_BOUND, HIGH_BOUND)

if args.eval:
    model.load()
    model.evaluate(render=True)
    sys.exit(0)


def train():

    rewards = []
    nb_steps = []
    nb_total_steps = 0
    time_beginning = time.time()

    try:

        for episode in range(Config.MAX_EPISODES):

            done = False
            step = 0
            episode_reward = 0

            state = env.reset()

            while not done and step < Config.MAX_STEPS:

                action = model.select_action(state)

                # Add noise
                noise = np.random.normal(scale=Config.EXPLO_SIGMA)
                action = np.clip(action+noise, LOW_BOUND, HIGH_BOUND)

                # Perform an action
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                # Save transition into memory
                model.memory.push(state, action, reward, next_state, 1-int(done))
                state = next_state

                model.optimize()

                step += 1
                nb_total_steps += 1

            print(f"Total T: {nb_total_steps}, "
                  f"Episode Num: {episode}, "
                  f"Episode T: {step}, "
                  f"Reward: {episode_reward}")

            nb_steps.append(step)
            rewards.append(episode_reward)

    except KeyboardInterrupt:
        pass

    finally:
        model.save()

    time_execution = time.time() - time_beginning

    print('---------------------------------------------------')
    print('---------------------STATS-------------------------')
    print('---------------------------------------------------')
    print(nb_total_steps, ' steps and updates of the network done')
    print(Config.MAX_EPISODES, ' episodes done')
    print('Execution time ', round(time_execution, 2), ' seconds')
    print('---------------------------------------------------')
    print('Average nb of steps per second : ', round(nb_total_steps/time_execution, 3), 'steps/s')
    print('Average duration of one episode : ', round(time_execution/Config.MAX_EPISODES, 3), 's')
    print('---------------------------------------------------')

    plt.figure()
    plt.plot(rewards)

    plt.savefig('results.png')
    # plt.show()


if __name__ == '__main__':
    train()
