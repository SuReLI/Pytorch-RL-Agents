import sys
sys.path.extend(['../commons/'])

import threading
import socket

from utils import ReplayMemory, str_to_list


class RemoteMemory(ReplayMemory):

    def listen(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.thread = threading.Thread(target=self.thread_listen)
        self.thread.start()

    def thread_listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('localhost', 12800))
        sock.listen(5)
        connection, address = sock.accept()

        with connection:
            msg = ''

            while True:
                data = connection.recv(8192)
                if not data:
                    break
                msg += data.decode()

                if '\n' in msg:
                    exp_sample, msg = msg.split('\n', 1)
                    self.compute(exp_sample)

        sock.close()

    def compute(self, msg):
        msg = msg.split(' ; ')
        s = str_to_list(msg[0])
        a = str_to_list(msg[1])
        r = float(msg[2])
        s_ = str_to_list(msg[3])
        d = int(msg[4])
        self.push(s, a, r, s_, d)

    def close(self):
        self.thread.join()
