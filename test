#!/usr/bin/env python

import argparse

from agents.DDPG.model import DDPG
from agents.DQN.model import DQN
from agents.SAC.model import SAC
from agents.TD3.model import TD3
from commons.run_expe import test


parser = argparse.ArgumentParser(description='Test an agent in a gym environment')
parser.add_argument('agent', nargs='?', default='DDPG',
                    help="Choose the agent to train (one of {DDPG, TD3, SAC, DQN}).")
parser.add_argument('--no_render', action='store_false', dest="render",
                    help='Display the tests')
parser.add_argument('-n', '--nb_tests', default=10, type=int, dest="nb_tests",
                    help="Number of evaluation to perform.")
parser.add_argument('--gif', action='store_true', dest="gif",
                    help='Save a gif of a test')
parser.add_argument('-f', '--folder', default=None, type=str, dest="folder",
                    help="Folder where the models are saved")
args = parser.parse_args()

if args.agent == 'DDPG':
    agent = DDPG

elif args.agent == 'TD3':
    agent = TD3

elif args.agent == 'SAC':
    agent = SAC

elif args.agent == 'DQN':
    agent = DQN

else:
    raise Exception("Agent invalid")

test(agent, args)
