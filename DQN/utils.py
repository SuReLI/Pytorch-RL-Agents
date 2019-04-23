import os

import math
from collections import namedtuple
import numpy as np

from PIL import Image
import torchvision.transforms as T



Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


def get_epsilon_threshold(episodes, params):
    """Returns the epsilon value (for exploration) following an exponential
    decreasing. """
    return params['EPSILON_END'] + (params['EPSILON_START'] - params['EPSILON_END']) * \
        math.exp(-1 * episodes / params['EPSILON_DECAY'])


def update_targets(target, original, tau):
    """Soft update from `original` network to `target` network with a rate TAU
    (defined in parameters.py) """
    for targetParam, orgParam in zip(target.parameters(), original.parameters()):
        targetParam.data.copy_((1 - tau)*targetParam.data + tau*orgParam.data)


def update_shared_targets(source, dest, tau, id_last_layer=None):
    """Soft update from `source` network to `dest` network with a rate TAU
    (defined in parameters.py) """
    for src_params, dst_params in zip(source.phi.parameters(), dest.phi.parameters()):
        dst_params.data.copy_((1 - tau)*dst_params.data + tau*src_params.data)

    if id_last_layer is None:
        for src_params, dst_params in zip(source.last_layers.parameters(), dest.last_layers.parameters()):
            dst_params.data.copy_((1 - tau)*dst_params.data + tau*src_params.data)

    else:
        for src_params, dst_params in zip(source.last_layers[id_last_layer].parameters(), dest.last_layers[id_last_layer].parameters()):
            dst_params.data.copy_((1 - tau)*dst_params.data + tau*src_params.data)


def ask_loading(file, load):
	print(file)
	if load and os.path.isfile(file):
		choice = input("A saved model has been found. Do you want to load it ? [y/N]")
		return (len(choice) > 0 and choice.lower()[:1] == 'y')
	return False