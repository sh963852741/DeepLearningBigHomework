
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
import pickle

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow
from tensorflow import keras
# from tensorflow import keras,losses
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import losses
from tensorflow.keras import optimizers

raw_dataframe = pandas.read_csv(".\\data\\pue.csv", encoding="gb2312")
raw_dataframe.dropna(inplace=True)
# deleted_raw_dataframe = raw_dataframe.drop(raw_dataframe.columns[:106], axis=1)
y_dataframe = raw_dataframe.pop(raw_dataframe.columns[-1])
# 删除时间戳
# raw_dataframe.pop(raw_dataframe.columns[-1])
# raw_numpy = raw_dataframe.to_numpy()
raw_numpy = MinMaxScaler().fit_transform(raw_dataframe, y_dataframe)
# 前64个维度不参与筛选
X_numpy_noselect = raw_numpy[:, :64]
X_numpy_toSelect = raw_numpy[:, 64:]
y_numpy = y_dataframe.to_numpy().reshape((-1))
X_numpy_selected = SelectKBest(f_regression, k=300-64).fit_transform(X=X_numpy_toSelect,y=y_numpy)
y_numpy = y_numpy.reshape((-1, 1))
X_numpy = np.hstack((X_numpy_noselect, X_numpy_selected))

train_X_numpy, test_X_numpy, train_y_numpy, test_y_numpy = train_test_split(X_numpy, y_numpy, test_size=0.2, shuffle=False)

model = Sequential(
    [
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ]
)

model.compile(optimizer= optimizers.Adam(0.001),
              loss=losses.mse,
              metrics='MeanSquaredError')

model.fit(train_X_numpy, train_y_numpy, epochs=30, batch_size=32)
model.evaluate(test_X_numpy, test_y_numpy, verbose=2)

# model.evaluate(test_X_numpy, test_y_numpy, verbose=2)

def draw_picture(input, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(input, training=False)
    plot.plot([x for x in range(0, len(predictions))], predictions, linestyle="dashed")
    plot.plot([x for x in range(0, len(predictions))], labels)
    plot.show()

draw_picture(test_X_numpy, test_y_numpy)

model.save('.\\dist\\fnn_model.h5')