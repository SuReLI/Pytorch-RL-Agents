import socket
import gym

import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


env = gym.make(config["GAME"])

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect(('localhost', 12800))
    while True:
        s = env.reset()
        a = env.action_space.sample()
        s_, r, d, _ = env.step(a)
        data = [list(s), list(a), r, list(s_), 1-int(d)]
        data = ' ; '.join(map(str, data)) + '\n'
        print("Send data : ", data)
        sock.sendall(data.encode())
        try:
            input()
        except KeyboardInterrupt:
            break
    sock.close()
