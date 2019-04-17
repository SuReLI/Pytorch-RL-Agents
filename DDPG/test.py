

parser = argparse.ArgumentParser(description='Run DDPG on ' + Config.GAME)
parser.add_argument('-n', action='store_true', help='Use GPU')

env = gym.make(Config.GAME)

print(env.action_space)

LOW_BOUND = int(env.action_space.low[0])
HIGH_BOUND = int(env.action_space.high[0])
STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.shape[0]

print("Creating neural networks and optimizers...")
model = Model(device, STATE_SIZE, ACTION_SIZE, LOW_BOUND, HIGH_BOUND)

model.load()
model.evaluate(render=True)
sys.exit(0)