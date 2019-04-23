import os

import math
from collections import namedtuple
import numpy as np

from PIL import Image
import torchvision.transforms as T



Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

SuperTransition = namedtuple('SuperTransition', ('state', 'qvalues'))

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen(env, device):
    """Get the image from an environment to learn from pixels"""
    # transpose into torch order CHW
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


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

#copy phi from input_net and load into output_net
def copy_phi(input_net,output_net):
    output_net.state_dict()['hidden1.weight'].data.copy_(input_net.hidden1.weight)
    output_net.state_dict()['hidden1.bias'].data.copy_(input_net.hidden1.bias)
    output_net.state_dict()['hidden2.weight'].data.copy_(input_net.hidden2.weight)
    output_net.state_dict()['hidden2.bias'].data.copy_(input_net.hidden2.bias)


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


class Converter:

    def __init__(self, envs):
        action_sizes = [0] + [env.observation_space.shape[0] for env in envs]
        self.idx_states = np.cumsum(action_sizes)
        self.total_size = sum(action_sizes)
        self.n_games = len(envs)

    def __call__(self, state, id_game):
        assert id_game < self.n_games

        new_state = np.zeros(self.total_size)
        new_state[self.idx_states[id_game]:self.idx_states[id_game+1]] = state
        return new_state

def ask_loading(file, load):
	print(file)
	if load and os.path.isfile(file):
		choice = input("A saved model has been found. Do you want to load it ? [y/N]")
		return (len(choice) > 0 and choice.lower()[:1] == 'y')
	return False