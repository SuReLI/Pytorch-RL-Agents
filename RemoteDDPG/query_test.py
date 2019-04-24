import socket
import gym
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

env = gym.make(config["GAME"])

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect(('localhost', 12801))
    while True:
        s = env.reset()
        data = str(list(s)) + '\n'
        print("Send state : ", str(list(s)))
        sock.sendall(data.encode())
        ans = sock.recv(8192)
        print("Action received : ", ans.decode())
        try:
            input()
        except KeyboardInterrupt:
            break
    sock.close()

