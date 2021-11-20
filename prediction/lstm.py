import numpy as np
import pandas
from sklearn.metrics import r2_score
import matplotlib.pyplot as plot

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow
import tensorflow.keras as keras
import tensorflow.keras.models
from tensorflow.python.keras.backend import dropout

BATCH_SIZE = 16

raw_dataframe = pandas.read_csv(".\\data\\pue.csv", encoding="gb2312") # .iloc[240:, :]
raw_dataframe.dropna(inplace=True)
# px =raw_dataframe.iloc[:,-2].to_numpy()
# py =raw_dataframe.iloc[:,-1].to_numpy()
# plot.plot(px, py)
# plot.show()

y_dataframe = raw_dataframe.pop(raw_dataframe.columns[-1])

# raw_numpy = raw_dataframe.to_numpy()
# raw_numpy = StandardScaler().fit_transform(raw_dataframe, y_dataframe)
raw_numpy = MinMaxScaler().fit_transform(raw_dataframe)
# 前64个维度不参与筛选
X_numpy_noselect = raw_numpy[:, :64]
X_numpy_toSelect = raw_numpy[:, 64:]
y_numpy = y_dataframe.to_numpy().reshape((-1))
X_numpy_selected = SelectKBest(
    f_regression, k=128-64).fit_transform(X=X_numpy_toSelect, y=y_numpy)
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
    # return data, labels
    return np.array(data), np.array(labels)


X_numpy, y_numpy = multivariate_data(X_numpy, y_numpy, 4)
train_X_numpy, test_X_numpy, train_y_numpy, test_y_numpy = train_test_split(X_numpy, y_numpy, test_size=0.2, shuffle=False)

train_data_single = tensorflow.data.Dataset.from_tensor_slices((train_X_numpy, train_y_numpy))
train_data_single = train_data_single.cache().batch(BATCH_SIZE)#.repeat().shuffle(10000)
val_data_single = tensorflow.data.Dataset.from_tensor_slices((test_X_numpy, test_y_numpy))
val_data_single = val_data_single.batch(BATCH_SIZE)#.repeat()
# ppp = train_X_numpy.shape[-2:]
single_step_model = keras.models.Sequential()
single_step_model.add(keras.layers.Bidirectional(keras.layers.LSTM(256, input_shape=train_X_numpy.shape[-2:], dropout=0.1)))
single_step_model.add(keras.layers.Dense(1))
# single_step_model.add(keras.layers.Dropout(0.5))

single_step_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=keras.losses.mean_squared_error)

single_step_history = single_step_model.fit(
    train_data_single, epochs=50, validation_data=val_data_single)

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plot.figure()
    plot.plot(epochs, loss, 'b', label='Training loss')
    plot.plot(epochs, val_loss, 'r', label='Validation loss')
    plot.title(title)
    plot.legend()
    plot.show()

# plot_train_history(single_step_history, "")

res = single_step_model.predict(val_data_single)
p = res.reshape((-1))
plot.plot([x for x in range(0, len(test_y_numpy))],test_y_numpy, 'r', label='real')
plot.plot([x for x in range(0, len(test_y_numpy))],p,'b', label='predict')
plot.legend()
plot.show()

score = r2_score(test_y_numpy, p, multioutput='raw_values')
print(f'{type(single_step_model)}回归模型R2分数(按列原始输出)：'+str(score))