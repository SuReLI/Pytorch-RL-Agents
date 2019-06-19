#!/usr/bin/env python

import argparse

from agents.DDPG.model import DDPG
from agents.DQN.model import DQN
from agents.SAC.model import SAC
from agents.TD3.model import TD3
from commons.run_expe import train

parser = argparse.ArgumentParser(description='Train an agent in a gym environment')
parser.add_argument('agent', nargs='?', default='DDPG',
                    help="Choose the agent to train (one of {DDPG, TD3, SAC, DQN}).")
parser.add_argument('--no_gpu', action='store_false', dest='gpu', help="Don't use GPU")
parser.add_argument('--load', dest='load', type=str, help="Load model")
args = parser.parse_args()


if args.agent == 'DDPG':
    agent = DDPG

elif args.agent == 'TD3':
    agent = TD3

elif args.agent == 'SAC':
    agent = SAC

elif args.agent == 'DQN':
    agent = DQN

train(agent, args)
