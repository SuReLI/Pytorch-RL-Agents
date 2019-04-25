import os
import time
from datetime import datetime
import argparse
try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

import numpy as np
import gym
import yaml

import torch
from tensorboardX import SummaryWriter

from model import Model

import utils

print("DQN starting...")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description='Run DQN on ' + config["GAME"])
parser.add_argument('--gpu', help='Use GPU', action='store_true')
args = parser.parse_args()

# Create folder and writer to write tensorboard values
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
folder = f'runs/{current_time}_{config["GAME"][:4]}'
writer = SummaryWriter(folder)
if not os.path.exists(folder+'/models/'):
    os.mkdir(folder+'/models/')

# Write optional info about the experiment to tensorboard
for k, v in config.items():
    writer.add_text('Config',  str(k) + ' : ' + str(v), 0)

with open(folder+'/config.yaml', 'w') as file:
    yaml.dump(config, file)

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
env = gym.make(config["GAME"])

STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n

def train():

    print("Creating neural networks and optimizers...")
    model = Model(device, STATE_SIZE, ACTION_SIZE, folder, config)

    nb_total_steps = 0
    time_beginning = time.time()

    try:
        print("Starting training...")
        nb_episodes_done = 0
        steps_per_sec = []

        for episode in range(config['MAX_EPISODES']) :
            time_beginning_ep = time.time()

            # Initialize the environment and state
            state = env.reset()
            state = torch.tensor([state], dtype=torch.float, device=model.device)
            episode_reward = 0
            done = False
            step = 0

            while step <= config['MAX_TIMESTEPS'] and not done:

                # Select and perform an action
                action = model.select_action(state, episode)
                next_state, reward, done, _ = env.step(action.item())
                next_state = torch.tensor([next_state], dtype=torch.float, device=model.device)
                reward = model.intermediate_reward(reward, next_state)
                episode_reward += reward.item()

                if not done and step == config['MAX_TIMESTEPS']:
                    done = True

                if done : next_state = None

                # Store the transition in memory
                model.memory.push(state, action, reward, next_state)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                loss = model.optimize_model()

                step += 1
                nb_total_steps += 1

            # Update the target network
            if episode % model.config['TARGET_UPDATE'] == 0:
                utils.update_targets(model.agent.target_nn, model.agent.nn, model.config['TAU'])
            nb_episodes_done += 1

            # Write scalars to tensorboard
            writer.add_scalar('stats/reward_per_episode', episode_reward, episode)
            if loss is not None:
                writer.add_scalar('stats/loss', loss, episode)

            # Write info to terminal
            steps_per_sec.append(round(step/(time.time() - time_beginning_ep),3))
            if episode % 100 == 0 :
                print(f'Episode {episode}, Reward: {round(episode_reward, 2)}, '
                      f'Steps: {step}, Epsilon: {utils.get_epsilon_threshold(nb_episodes_done, model.config):.5}, '
                      f'LR: {model.agent.optimizer.param_groups[0]["lr"]:.4f}, Speed : {round(sum(steps_per_sec[-20:])/20,1) } steps/s')


    except KeyboardInterrupt:
        pass

    finally:
        model.save()
        print("\n\033[91m\033[1mModel saved in", folder, "\033[0m")

    time_execution = time.time() - time_beginning

    print('\n---------------------STATS-------------------------\n',
          nb_total_steps, ' steps and updates of the network done\n',
          nb_episodes_done, ' episodes done\n'
          'Execution time : ', round(time_execution, 2), ' seconds\n'
          'Average nb of steps per second : ', round(nb_total_steps/time_execution, 3), 'steps/s\n'
          'Average duration of one episode : ', round(time_execution/nb_episodes_done, 3), 's\n')

    writer.close()


if __name__ == '__main__':
    train()
