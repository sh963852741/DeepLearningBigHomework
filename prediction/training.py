import numpy as np
import pandas
import torch
import torch.nn
import torch.optim
import torch.nn.modules
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
torch.set_default_tensor_type(torch.DoubleTensor)

from model import PUEForecast
from pue_dataset import PUEDataset
raw_dataframe = pandas.read_csv(".\\data\\pue.csv", encoding="gb2312").iloc[240:, :]
raw_dataframe.dropna(inplace=True)
# deleted_raw_dataframe = raw_dataframe.drop(raw_dataframe.columns[:106], axis=1)
y_dataframe = raw_dataframe.pop(raw_dataframe.columns[-1])
# 删除时间戳
# raw_dataframe.pop(raw_dataframe.columns[-1])
# raw_numpy = raw_dataframe.to_numpy()
raw_numpy = StandardScaler().fit_transform(raw_dataframe, y_dataframe)
# 前64个维度不参与筛选
X_numpy_noselect = raw_numpy[:, :64]
X_numpy_toSelect = raw_numpy[:, 64:]
y_numpy = y_dataframe.to_numpy().reshape((-1))
X_numpy_selected = SelectKBest(f_regression, k=128-64).fit_transform(X=X_numpy_toSelect,y=y_numpy)
y_numpy = y_numpy.reshape((-1, 1))
X_numpy = np.hstack((X_numpy_noselect, X_numpy_selected))

train_X_numpy, test_X_numpy, train_y_numpy, test_y_numpy = train_test_split(X_numpy, y_numpy, test_size=0.2, shuffle=False)
# test_X_numpy = np.random.rand(455, 1024) * 100
train_dataset = PUEDataset(
    torch.from_numpy(train_X_numpy), torch.from_numpy(train_y_numpy))
train_dataloader = DataLoader(train_dataset, batch_size=16)
test_dataset = PUEDataset(
    torch.from_numpy(test_X_numpy), torch.from_numpy(test_y_numpy))
test_dataloader = DataLoader(test_dataset, batch_size=16)

model = PUEForecast()

# loss_fn = torch.nn.SmoothL1Loss()

loss_fn1 = torch.nn.CrossEntropyLoss(label_smoothing = 0.1)
loss_fn2 = torch.nn.SmoothL1Loss(label_smoothing = 0.1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train(dataloader: DataLoader, model: torch.nn.Module, loss_fn1: _Loss, loss_fn2: _Loss, optimizer: torch.optim.Optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, target_in, target_out, target_out_pue) in enumerate(dataloader):
        # X, y = X.to(device), y.to(device)

        # X, shape[batch_size, 有几行, 每行有多少特征]
        # target_in, shape[batch_size, 2, 1]


        # target_out, shape[batch_size, 2, 128] pue之前的特征值
        # target_out_pue, shape[batch_size, 2, 1] pue值

        pred, pue = model(X, target_in)

        loss1 = loss_fn1(pred, target_out)
        loss2 = loss_fn2(pue,target_out_pue)
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

train(train_dataloader, model, loss_fn1, loss_fn2, optimizer)

def test(dataloader: DataLoader, model: torch.nn.Module, loss_fn: _Loss):
    plot.figure()
    y_axis_predict = []
    y_axis_real = []

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, np.nan
    with torch.no_grad():
        for X, target_in, target_out in dataloader:
            # X, y = X.to(device), y.to(device)

            # pred, shape:[batch_size, (BOS+pue=2), 1]
            pred, pue = model(X, target_in)
            test_loss += loss_fn(pue, target_out).item()

            for i in range(0, pred.shape[0]):
                y_axis_predict.append(pred[i][0][0])
                y_axis_real.append(target_out[i][0][0])
                print(f"predict: {pred[i][0][0]}\treal: {target_out[i][0][0]}")
    test_loss /= num_batches
    # correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    plot.plot([x for x in range(0, len(y_axis_real))], y_axis_predict, linestyle="dashed")
    plot.plot([x for x in range(0, len(y_axis_real))], y_axis_real)
    plot.show()

test(test_dataloader, model, loss_fn2)

torch.save(model, ".\\dist\\neural_network_model.pkl") 