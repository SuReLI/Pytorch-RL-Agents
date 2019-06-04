import os
import signal
import traceback
import argparse
from datetime import datetime
from tqdm import trange

import gym
import yaml
import torch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from model import Model
from utils import NormalizedActions


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description='Run SAC on ' + config["GAME"])
parser.add_argument('--no_gpu', action='store_false', dest='gpu', help="Don't use GPU")
args = parser.parse_args()


if not os.path.exists('runs'):
    os.mkdir('runs')

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
folder = f'runs/{config["GAME"].split("-")[0]}_{current_time}'
writer = SummaryWriter(folder)
if not os.path.exists(folder+'/models/'):
    os.mkdir(folder+'/models/')

# Choose device cpu or cuda if a gpu is available
if args.gpu and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("\033[91m\033[1mDevice : ", device.upper(), "\033[0m")
writer.add_text('Device', device, 0)
device = torch.device(device)

writer.add_text('Game', config["GAME"], 0)

for k, v in config.items():
    if not k.startswith("__"):
        writer.add_text('Config', str(k) + ' : ' + str(v), 0)

with open(folder+'/config.yaml', 'w') as file:
    yaml.dump(config, file)


env = NormalizedActions(gym.make(config["GAME"]))

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

print("Initializing Networks...")
model = Model(state_size, action_size, device, folder, config)


def handler(sig, frame):
    # model.evaluate(n_ep=1, render=True)
    print("Can't evaluate right now !")


signal.signal(signal.SIGTSTP, handler)


def train():

    total_step = 0
    rewards = []
    eval_rewards = []
    lenghts = []

    try:
        for episode in trange(config["MAX_EPISODES"]):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_len = 0

            for step in range(config["MAX_STEPS"]):
                if total_step > 10000:
                    action = model.soft_actor.get_action(state)
                else:
                    action = env.action_space.sample()

                next_state, reward, done, _ = env.step(action)

                model.memory.push(state, action, reward, next_state, done)
                model.optimize()

                state = next_state
                episode_reward += reward
                total_step += 1
                episode_len += 1

                if done:
                    break

            rewards.append(episode_reward)
            eval_rewards.append(model.evaluate())
            lenghts.append(episode_len)

            if episode % config["FREQ_EVAL"] == 0:
                plt.figure()
                plt.cla()
                plt.plot(rewards)
                plt.savefig(folder + '/rewards.png')
                plt.close()

                plt.figure()
                plt.cla()
                plt.plot(eval_rewards)
                plt.savefig(folder + '/eval_rewards.png')
                plt.close()

                plt.figure()
                plt.cla()
                plt.plot(lenghts)
                plt.savefig(folder + '/lenghts.png')
                plt.close()

    except KeyboardInterrupt:
        pass

    except Exception:
        print(traceback.format_exc())

    finally:
        model.save()


if __name__ == '__main__':
    train()
