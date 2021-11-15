import torch
from torch.utils.data import Dataset
import numpy

BOS = numpy.ones
EOS = numpy.zeros


class PUEDataset(Dataset):
    def __init__(self, X: numpy.ndarray, y: numpy.ndarray, src_size: int = 8):
        if(X.shape[0] != y.shape[0]):
            raise Exception("长度不一致")
        self.source = X
        self.target = y
        self.src_size = src_size

    def __len__(self):
        return self.source.shape[0] - self.src_size + 1 - 1

    def __getitem__(self, index):
        # shape:[有多少条数据, 每个数据有多少列]
        source = self.source[index: index + self.src_size]
        # shape:[1, 1088]
        target_in = self.source[index + self.src_size: index + self.src_size + 1]
        # target_in = numpy.insert(target_in, 0, BOS((1024)), axis=0)  # shape:[2, 1088]
        target_out = self.target[index + self.src_size: index + self.src_size + 1]
        # target_out = numpy.insert(target_out, 1, EOS((1)), axis=0)  # shape:[2, 1]
        return torch.tensor(source), torch.tensor(target_in), torch.tensor(target_out)
