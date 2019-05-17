import sys
sys.path.extend(['../commons/'])

import os
import argparse
import yaml
import gym
import torch

from model import Model
from utils import get_latest_dir

parser = argparse.ArgumentParser(description='Test DDPG')
parser.add_argument('--no_render', action='store_false', dest="render",
                    help='Display the tests')
parser.add_argument('-n', '--nb_tests', default=10, type=int, dest="nb_tests",
                    help="Number of evaluation to perform.")
parser.add_argument('-f', '--folder', default=None, type=str, dest="folder",
                    help="Folder where the models are saved")
args = parser.parse_args()

if args.folder is None:
    args.folder = os.path.join('runs/', get_latest_dir('runs/'))

with open(os.path.join(args.folder, 'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

device = torch.device('cpu')

# Create gym environment
env = gym.make(config["GAME"])

STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n

# Creating neural networks and loading models
model = Model(device, STATE_SIZE, ACTION_SIZE, args.folder, config)
model.load()

score = model.evaluate(n_ep=args.nb_tests, render=args.render)
print(f"Average score : {score}")
