import sys
sys.path.extend(['../commons/'])

import os
import argparse
import yaml
import gym
import gym_hypercube
try:
    import roboschool
except ModuleNotFoundError:
    pass
import torch

from model import Model
from utils import get_latest_dir, NormalizedActions


parser = argparse.ArgumentParser(description='Test SAC')
parser.add_argument('--no_render', action='store_false', dest="no_render",
                    help='Display the tests')
parser.add_argument('--gif', action='store_true', dest="gif",
                    help='Save a gif of a test')
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
env = NormalizedActions(gym.make(config["GAME"], n_dimensions=1, acceleration=False))

# Creating neural networks and loading models
model = Model(device, args.folder, config)
model.load()
print("\033[91m\033[1mModel loaded from ", args.folder, "\033[0m")

score = model.evaluate(n_ep=args.nb_tests, render=args.no_render)
model.plot_Q(pause=True)

print(f"Average reward : {score}")
