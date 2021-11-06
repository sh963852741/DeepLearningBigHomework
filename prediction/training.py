import pandas
import torch
import torch.nn
import torch.optim
import torch.nn.modules
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
torch.set_default_tensor_type(torch.DoubleTensor)

from model import PUEForecast
from pue_dataset import PUEDataset
raw_dataframe = pandas.read_csv(".\\data\\pue.csv", encoding="gb2312")
# deleted_raw_dataframe = raw_dataframe.drop(raw_dataframe.columns[:106], axis=1)
y_dataframe = raw_dataframe.pop(raw_dataframe.columns[-1])
X_dataframe = raw_dataframe
X_numpy = X_dataframe.to_numpy()
y_numpy = y_dataframe.to_numpy().reshape((-1, 1))
print(type(X_numpy))
train_dataset = PUEDataset(
    torch.from_numpy(X_numpy), torch.from_numpy(y_numpy))
train_dataloader = DataLoader(train_dataset, batch_size=4)

model = PUEForecast()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader: DataLoader, model: torch.nn.Module, loss_fn: _Loss, optimizer: torch.optim.Optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, target_in, target_out) in enumerate(dataloader):
        # X, y = X.to(device), y.to(device)

        # X, shape[batch_size, 有几行, 每行有多少特征]
        # target_in, shape[batch_size, 2, 1]
        # target_out, shape[batch_size, 2, 1]
        pred = model(X, target_in, target_out)
        loss = loss_fn(pred, target_in)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

train(train_dataloader, model, loss_fn, optimizer)
