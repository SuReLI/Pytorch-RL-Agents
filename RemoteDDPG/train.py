import sys
sys.path.extend(['../commons/'])

import os
import time
from datetime import datetime
import argparse
try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

import yaml

import torch
from tensorboardX import SummaryWriter

from RemoteMemory import RemoteMemory as ReplayMemory
from model import Model


print("DDPG starting...")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser(description='Run DDPG on ' + config["GAME"])
parser.add_argument('--no-gpu', action='store_true', dest='no_gpu', help="Don't use GPU")
args = parser.parse_args()

# Create folder and writer to write tensorboard values
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
expe_name = f'runs/{config["GAME"].split("-")[0]}_{current_time}'
writer = SummaryWriter(expe_name)
if not os.path.exists(expe_name+'/models/'):
    os.mkdir(expe_name+'/models/')

# Write optional info about the experiment to tensorboard
for k, v in config.items():
    writer.add_text('Config', str(k) + ' : ' + str(v), 0)

with open(expe_name+'/config.yaml', 'w') as file:
    yaml.dump(config, file)

# Choose device cpu or cuda if a gpu is available
if not args.no_gpu and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("\033[91m\033[1mDevice : ", device.upper(), "\033[0m")
writer.add_text('Device', device, 0)
device = torch.device(device)


LOW_BOUND = -1
HIGH_BOUND = 1
STATE_SIZE = 24
ACTION_SIZE = 4


def train():

    print("Creating neural networks and optimizers...")
    memory = ReplayMemory(config["MEMORY_CAPACITY"])
    memory.listen(STATE_SIZE, ACTION_SIZE)

    model = Model(device, memory, STATE_SIZE, ACTION_SIZE, LOW_BOUND, HIGH_BOUND, expe_name, config)
    model.query()

    time_beginning = time.time()

    try:
        print("Starting training...")
        step = 0
        while step < 100000:
            actor_loss, critic_loss = model.optimize()

            if actor_loss is not None:
                writer.add_scalar('loss/actor_loss', actor_loss, step)
                writer.add_scalar('loss/critic_loss', critic_loss, step)

                step += 1

    except KeyboardInterrupt:
        pass

    except Exception as e:
        print("Exception", e)

    finally:
        memory.close()
        writer.close()
        model.close()
        model.save()
        print("\033[91m\033[1mModel saved in", expe_name, "\033[0m")

    time_execution = time.time() - time_beginning

    print('---------------------------------------------------\n'
          '---------------------STATS-------------------------\n'
          '---------------------------------------------------\n',
          step, ' steps and updates of the network done\n',
          'Execution time ', round(time_execution, 2), ' seconds\n'
          '---------------------------------------------------\n'
          'Average nb of steps per second : ', round(step/time_execution, 3), 'steps/s\n'
          '---------------------------------------------------')


if __name__ == '__main__':
    train()
