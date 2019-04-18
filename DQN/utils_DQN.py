import os

import math
from collections import namedtuple

from parameters import Parameters


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


def get_epsilon_threshold(episodes, params):
    """Returns the epsilon value (for exploration) following an exponential
    decreasing. """
    return params['EPSILON_END'] + (params['EPSILON_START'] - params['EPSILON_END']) * \
        math.exp(-1 * episodes / params['EPSILON_DECAY'])


def ask_loading(file):
	print(file)
	if Parameters.LOAD and os.path.isfile(file):
		choice = input("A saved model has been found. Do you want to load it ? [y/N]")
		return (len(choice) > 0 and choice.lower()[:1] == 'y')
	return False