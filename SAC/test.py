import os
import argparse
import yaml

import gym
import torch

from model import Model
from utils import get_latest_dir, NormalizedActions


parser = argparse.ArgumentParser(description='Test SAC')
parser.add_argument('--no_render', action='store_false', dest="no_render",
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

env = NormalizedActions(gym.make(config["GAME"]))

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

model = Model(state_size, action_size, device, args.folder, config)
model.load()


score = model.evaluate(n_ep=args.nb_tests, render=args.no_render)

print(f"Average reward : {score}")
