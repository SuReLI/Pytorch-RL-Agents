# DDPG

This repository contains an open source implementation of the DDPG algorithm presented in [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971).

## Requirements
* Pytorch
* gym
* tensorboardX

## Training the agent

The configuration of your experiement must be specified in the ```config.yaml``` file at the root of the repository.

To train an agent to play, simply run : ```python3 train.py```.

The models will be saved in a folder in ```runs/```.

## Visualizing the training

You can use Tensorboard to visualize the evolution of the training in real time :

1. Open a new terminal
2. Run ```tensorboard --logdir=<absolute-path>/runs/```.
3. In your browser navigate to localhost:6006/

## Testing the agent

To test the performance of a trained agent, run ```python3 test.py -f runs/YOUR_FOLDER```.

Options :
* ```--no_render``` : The tests will not be graphically displayed.
* ```-n X``` : Specify the number X of episodes you want to play.
* ```--gif``` : A gif of the test will be saved.s