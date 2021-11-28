import pickle
import re
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
        self.current_position = 2083 # pue = 1.586454183
        self.current_row = self.X[self.current_position, :]
        # self.current_action = self.current_row[:64]

    def reset(self):
        self.current_row = self.X[self.current_position, :]
        # self.current_action = self.current_row[:64]
        # return np.delete(self.X[self.current_position], [x for x in range(0,64)])
        return self.current_row

    def set_row(self, row):
        self.current_position = row
        return self.y[self.current_position], self.model.predict([self.X[self.current_position, :]])[0]

    def step(self, action_change: np.ndarray) -> tuple([np.ndarray, float, bool]):
        # row = self.X[self.current_position]
        # row = np.delete(row, [x for x in range(0,64)])
        # self.current_action += self.current_action + action_change
        # row = np.hstack((self.current_action, row))
        row = action_change
        # zero = np.zeros(row.shape)
        pred = self.model.predict([row])
        reward = (self.y[self.current_position] - pred[0])
        reward = reward * 1000
        
        # reward += 308000
        # self.current_position += 1
        if(self.current_position == self.X.shape[0]):
            return None, reward, True, None, pred
        else:
            # return np.delete(self.X[self.current_position], [x for x in range(0,64)]), reward, False, None
            return row, reward, False, None, pred

