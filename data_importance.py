from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import xlrd
import numpy as np

# X, Y = make_regression(n_samples=10, n_features=4, n_informative=1, random_state=0, shuffle=False)

path_data = "data/airdata.xlsx"

data = xlrd.open_workbook(path_data)
table = data.sheet_by_index(0)
data_num = table.nrows - 1

hour = np.array(table.col_values(1)[1:])
pm25 = np.array(table.col_values(2)[1:])
o3 = np.array(table.col_values(3)[1:])
pm10 = np.array(table.col_values(4)[1:])
so2 = np.array(table.col_values(5)[1:])
no2 = np.array(table.col_values(6)[1:])
co = np.array(table.col_values(7)[1:])
temperature = np.array(table.col_values(8)[1:])
wind = np.array(table.col_values(9)[1:])
weather = np.array(table.col_values(10)[1:])
moisture = np.array(table.col_values(11)[1:])
pressure = np.array(table.col_values(12)[1:])
precipitation = np.array(table.col_values(13)[1:])

X = []
X.append(hour)
X.append(o3)
# X.append(pm10)
X.append(so2)
X.append(no2)
X.append(co)
X.append(temperature)
X.append(wind)
X.append(weather)
X.append(moisture)
X.append(pressure)
X.append(precipitation)
X = np.array(X)
X = np.transpose(X)
print(X.shape)

Y = np.array(pm25)
print(Y.shape)

regr = RandomForestRegressor().fit(X, Y)
print("RandomForestRegressor.feature_importances_:\n", regr.feature_importances_)

cca = CCA().fit(X, Y)
print("cca.x_weights_:\n", cca.x_weights_)
# print("cca.x_loadings_:\n", cca.x_loadings_)
# print("cca.x_scores_:\n", cca.x_scores_)
print("cca.score:\n", cca.score(X, Y))
#print("cca.predict:\n", cca.predict(X))
