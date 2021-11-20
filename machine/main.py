from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score
from sklearn import neighbors

from sklearn.decomposition import PCA

import numpy as np
import numpy.random as random
import pandas
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
pca = PCA(n_components=256-64)

raw_dataframe = pandas.read_csv(".\\data\\pue.csv", encoding="gb2312")
raw_dataframe.dropna(inplace=True)
# raw_dataframe.describe().to_csv(".\\dist\\describe.csv", encoding="gb2312")
# deleted_raw_dataframe = raw_dataframe.drop(raw_dataframe.columns[:106], axis=1)
y_dataframe = raw_dataframe.pop(raw_dataframe.columns[-1])
# 删除时间戳
# raw_dataframe.pop(raw_dataframe.columns[-1])
raw_numpy = raw_dataframe.to_numpy()
# raw_numpy = StandardScaler().fit_transform(raw_dataframe, y_dataframe)
# 前64个维度不参与筛选
X_numpy_noselect = raw_numpy[:, :64]
X_numpy_toSelect = raw_numpy[:, 64:]
y_numpy = y_dataframe.to_numpy().reshape((-1))

# X_numpy_selected = pca.fit_transform(X_numpy_toSelect)
X_numpy_selected = SelectKBest(f_regression, k=256-64).fit_transform(X=X_numpy_toSelect, y=y_numpy)
X_numpy = np.hstack((X_numpy_noselect, X_numpy_selected))
train_X_numpy, test_X_numpy, train_y_numpy, test_y_numpy = train_test_split(X_numpy, y_numpy, test_size=0.2, shuffle=False) # X_numpy[:-400], X_numpy[-400:], y_numpy[:-400], y_numpy[-400:]

np.savez(".\\data\\precessed_data.npz",x= X_numpy, y=y_numpy)
# test_X_numpy = random.rand(455, 256) * 100
knn = neighbors.KNeighborsRegressor(10, weights="uniform")
knn.fit(train_X_numpy, train_y_numpy)

from sklearn import ensemble
gbr = ensemble.GradientBoostingRegressor(n_estimators=100)
gbr.fit(train_X_numpy, train_y_numpy)

def use_model(model: RegressorMixin, X_test: np.ndarray, y_test: np.ndarray):
    y_forecast = model.predict(X_test)
    score = r2_score(y_test, y_forecast, multioutput='uniform_average')
    print(f'{type(model)}回归模型R2分数(按列直接平均)：'+str(score))
    score = r2_score(y_test, y_forecast, multioutput='variance_weighted')
    print(f'{type(model)}回归模型R2分数(按列方差加权平均)：'+str(score))
    score = r2_score(y_test, y_forecast, multioutput='raw_values')
    print(f'{type(model)}回归模型R2分数(按列原始输出)：'+str(score))
    
    plot.figure()
    plot.plot([x for x in range(0, len(y_forecast))], y_test)
    plot.plot([x for x in range(0, len(y_forecast))], y_forecast, linestyle="dashed")
    plot.show()

use_model(knn, test_X_numpy, test_y_numpy)
use_model(gbr, test_X_numpy, test_y_numpy)

import pickle
output = open('.\\dist\\machine_gbr.pkl', 'wb')
pickle.dump(gbr, output)
output.close()