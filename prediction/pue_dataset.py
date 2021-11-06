import torch
from torch.utils.data import Dataset
import numpy

BOS = [1]
EOS = [0]

class PUEDataset(Dataset):
    def __init__(self, X: numpy.ndarray, y: numpy.ndarray, src_size: int = 5):
        if(X.shape[0] != y.shape[0]):
            raise Exception("长度不一致")
        self.source = X
        self.target = y
        self.src_size = src_size

    def __len__(self):
        return self.source.shape[0] - self.src_size + 1

    def __getitem__(self, index):
        X = self.source[index : index + self.src_size] # shape:[有多少条数据, 每个数据有多少列]
        y = self.target[index + self.src_size -1 : index + self.src_size] # shape:[1, 1]
        target_in = numpy.insert(y, 0, BOS, axis=0) # shape:[2, 1]
        target_out = numpy.insert(y, 1, EOS, axis=0) # shape:[2, 1]
        return torch.tensor(X), torch.tensor(target_in), torch.tensor(target_out)
