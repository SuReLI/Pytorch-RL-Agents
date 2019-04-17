import argparse
# import roboschool
import gym
import torch
import yaml

from model import Model

with open('config.yaml', 'r') as stream:
    config = yaml.load(stream)

parser = argparse.ArgumentParser(description='Test DDPG on ' + config['GAME'])
parser.add_argument('--render', action='store_true', help='Display the tests', dest="render")
parser.add_argument('-n', '--nb_tests', default=10, type=int, help="Number of evaluation to perform.", dest="nb_tests")
parser.add_argument('-f', '--folder', default='runs/', type=str, help="Folder where the models are saved", dest="folder")
parser.add_argument('--gpu', action='store_true', help='Use GPU')
args = parser.parse_args()

# Choose device cpu or cuda if a gpu is available
if args.gpu and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("\033[91m\033[1mDevice : ", device.upper(), "\033[0m")
device = torch.device(device)

# Create gym environment
env = gym.make(config['GAME'])

LOW_BOUND = int(env.action_space.low[0])
HIGH_BOUND = int(env.action_space.high[0])
STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.shape[0]

# Creating neural networks and loading models
model = Model(device, STATE_SIZE, ACTION_SIZE, LOW_BOUND, HIGH_BOUND, args.folder)
model.load()

# START EVALUATION

rewards = [0]*args.nb_tests

for ep in range(args.nb_tests):

    state = env.reset()
    state = torch.tensor([state], dtype=torch.float, device=device)
    done = False
    step = 0
    while not done and step < config['MAX_STEPS']:

        action = model.actor(state).detach()
        state, r, done, _ = env.step(action.numpy()[0])
        if args.render:
            env.render()
        state = torch.tensor([state], dtype=torch.float, device=device)
        rewards[ep] += r
        step += 1

env.close()

print("Average reward : ", sum(rewards) / args.nb_tests)
