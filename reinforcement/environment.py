import pickle
from typing import Tuple
import numpy as np
class PUEEnviroment(object):
    def __init__(self, filepath = None):
        if(filepath != None):
            input = open(filepath, 'rb')
            model = pickle.load(input)
            self.model = model
            input.close()
        else:
            pass
        npzfile = np.load(".\\data\\precessed_data.npz")
        self.X = npzfile['x']
        self.y = npzfile['y']
        self.current_position = 0

    def reset(self):
        self.current_position = 0

    def step(self, action: np.ndarray) -> tuple([np.ndarray, float, bool]):
        row = self.X[self.current_position]
        row = np.delete(row, [x for x in range(0,64)])
        row = np.hstack((action, row))

        pred = self.model.fit(row)
        reward = self.y[self.current_position] - pred[0]
        self.current_position += 1
        if(self.current_position == self.X.shape[0]):
            return None, reward, True
        else:
            return np.delete(self.X[self.current_position], [x for x in range(0,64)]), reward, False
