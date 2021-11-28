import pickle
from typing import Tuple
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras import models


class PUEEnviroment(object):
    def __init__(self, filepath):
        self.model = models.load_model(filepath)
        self.current_row = 0
        npzfile = np.load(".\\data\\precessed_data.npz")
        self.X = npzfile['x']
        self.y = npzfile['y']

    def reset(self, row: int = 0):
        self.current_row = row
        array = np.zeros([64], np.uint64)
        array.fill(10)
        self.current_action = array
        return array

    def step(self, action_change: np.ndarray) -> tuple([np.ndarray, float, bool]):
        self.current_action = self.current_action + action_change
        reward = [x - self.current_action[x] for x in range(0, 64)]
        reward = np.sum(reward)

        # self.current_position += 1
        if(self.current_position == self.X.shape[0]):
            return None, reward, True, None
        else:
            return np.delete(self.X[self.current_position], [x for x in range(0, 64)]), reward, False, None
