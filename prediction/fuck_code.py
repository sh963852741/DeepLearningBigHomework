import numpy as np
import pandas

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow
import tensorflow.keras as keras
import tensorflow.keras.models

BATCH_SIZE = 16

raw_dataframe = pandas.read_csv(".\\data\\pue.csv", encoding="gb2312")
raw_dataframe.dropna(inplace=True)
y_dataframe = raw_dataframe.pop(raw_dataframe.columns[-1])
# raw_numpy = raw_dataframe.to_numpy()
raw_numpy = StandardScaler().fit_transform(raw_dataframe, y_dataframe)
# 前64个维度不参与筛选
X_numpy_noselect = raw_numpy[:, :64]
X_numpy_toSelect = raw_numpy[:, 64:]
y_numpy = y_dataframe.to_numpy()  # .reshape((-1))
X_numpy_selected = SelectKBest(
    f_regression, k=256-64).fit_transform(X=X_numpy_toSelect, y=y_numpy)
X_numpy = np.hstack((X_numpy_noselect, X_numpy_selected))


def multivariate_data(dataset, target, history_size):
    '''处理数据集

    Arguments:
    ----------
    - dataset：需要进行处理的数据
    - target：标签列
    - history_size：需要多少的历史数据
    '''
    data = []
    labels = []

    for i in range(history_size, len(dataset)):
        indices = range(i-history_size, i)
        data.append(dataset[indices])
        labels.append(target[i])

    return np.array(data), np.array(labels)


X_numpy, y_numpy = multivariate_data(X_numpy, y_numpy, 4)
train_X_numpy, test_X_numpy, train_y_numpy, test_y_numpy = train_test_split(X_numpy, y_numpy, test_size=0.2)

train_data_single = tensorflow.data.Dataset.from_tensor_slices((train_X_numpy, train_y_numpy))
train_data_single = train_data_single.batch(BATCH_SIZE)  # .repeat()
val_data_single = tensorflow.data.Dataset.from_tensor_slices((test_X_numpy, test_y_numpy))
val_data_single = val_data_single.batch(BATCH_SIZE)  # .repeat()
# ppp = train_X_numpy.shape[-2:]
single_step_model = keras.models.Sequential()
single_step_model.add(keras.layers.LSTM(32, input_shape=(train_X_numpy.shape[1], 1)))
single_step_model.add(keras.layers.Dense(1))

single_step_model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.MSE)

single_step_history = single_step_model.fit(
    train_X_numpy, train_y_numpy, batch_size=16, epochs=30)
