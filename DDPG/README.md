# DDPG

This repository contains an open source implementation of the DDPG algorithm presented in [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971).

## Requirements
* Pytorch
* gym
* tensorboardX

## Training the agent

The configuration of your experiement must be specified in the ```config.yaml``` file at the root of the repository.

To train an agent to play, simply run : ```python3 train.py```.

Options :
* ```--gpu``` : If available, the GPU will be used for computation.

The models will be saved in a folder in ```runs/```.

## Testing the agent

To test the performance of a trained agent, run ```python3 test.py -f runs/YOUR_FOLDER```.

Options :
* ```--gpu``` : If available, the GPU will be used for computation.
* ```--render``` : The tests will be graphically displayed.
* ```-n X``` : Specify the number X of episodes you want to play.