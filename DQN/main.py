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

# from model import Model
from Agent import Agent

import random
from utils import *

print("DQN starting...")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description='Run DQN on ' + config["GAME"])
parser.add_argument('--gpu', help='Use GPU', action='store_true')
args = parser.parse_args()

# Create folder and writer to write tensorboard values
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
expe_name = f'runs/{current_time}_{config["GAME"][:4]}'
writer = SummaryWriter(expe_name)
if not os.path.exists(expe_name+'/models/'):
    os.mkdir(expe_name+'/models/')

# Write optional info about the experiment to tensorboard
for k, v in config.items():
    writer.add_text('Config',  str(k) + ' : ' + str(v), 0)

with open(expe_name+'/config.yaml', 'w') as file:
    yaml.dump(config, file)

# Choose device cpu or cuda if a gpu is available
if args.gpu and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("\033[91m\033[1mDevice : ", device.upper(), "\033[0m")
writer.add_text('Device', device, 0)
device = torch.device(device)


if __name__ == "__main__":

    game = config['GAME']
    agent = Agent(device, game)

    print("Training !")

    agent.episodes_done = 0
    # stored_states = []
    # self.states = []
    states = []

    i_episode = 1
    agent.episodes_done = 1
    
    steps_per_sec = []

    while i_episode < agent.game_param['MAX_EPISODES'] : # and not agent.gui.STOP:

        time_beginning_ep = time.time()

        # Initialize the environment and state
        state = agent.env.reset()

        state = torch.tensor([state], dtype=torch.float, device=agent.device)

        # gui_render = agent.gui.render.get(i_episode)

        i_episode_reward = 0
        done = False
        step = 0

        while step <= agent.game_param['MAX_TIMESTEPS'] and not done:

            # Select and perform an action
            action = agent.select_action(state)
            next_state, reward, done, _ = agent.env.step(action.item())

            # Save the next state in a list, so we can store them in a csv file at the end
            # if not done and random.random() < agent.game_param['STATE_SAVE_PROBA']:
            #     stored_states.append(next_state)

            next_state = torch.tensor([next_state], dtype=torch.float, device=agent.device)

            reward = agent.intermediate_reward(reward, next_state)

            #accumulated reward for each episode
            i_episode_reward += reward.item()

            if done : next_state = None
            # Store the transition in memory
            agent.memory.push(state, action, reward, next_state)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            agent.optimize_model()

            step += 1

        writer.add_scalar('stats/reward_per_episode', i_episode_reward, i_episode)

        steps_per_sec.append(round(step/(time.time() - time_beginning_ep),3))
        if i_episode % 100 == 0 :
            print(f'Episode {i_episode}, Reward: {i_episode_reward}, '
                  f'Steps: {step}, Epsilon: {get_epsilon_threshold(agent.episodes_done, agent.game_param):.5}, '
                  f'LR: {agent.optimizer.param_groups[0]["lr"]:.4f}, average {round(sum(steps_per_sec[-20:])/20,1) } steps/s')

        # Update the target network
        if i_episode % agent.game_param['TARGET_UPDATE'] == 0:
            update_targets(agent.target_network, agent.policy_network, agent.game_param['TAU'])

        i_episode += 1
        agent.episodes_done += 1

    print('Complete')
