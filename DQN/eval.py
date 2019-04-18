import os
import sys
sys.path.extend(["../utils/", "../parameters/", "../results/DQN/"])

import torch

from Agent import Agent
from parameters import Parameters

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_game(args, default=2):
    if len(args) > 1:
        arg = args[1]
        if arg.isdigit() and 0 <= int(arg) < 3:
            return games[int(arg)]
        elif arg in games:
            return arg
        else:
            raise ValueError("This game is not recognized !")
    else:
        return default


if __name__ == "__main__":

    games = ['Acrobot-v1', 'MountainCar-v0', 'CartPole-v1']

    # Get game from command line argument
    game = get_game(sys.argv, default='CartPole-v1')

    print("Creating an agent that plays ", game)
    agent = Agent(device, game, play=True)

    agent.play()
