import numpy as np

np.random.seed(1337)  # for reproducibility
# from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
# import GridSearchCV

import xlrd
import math
import matplotlib.pyplot as plt

import os
import sys

path_DBN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep-belief-network")
sys.path.append(path_DBN)
from dbn.tensorflow import SupervisedDBNRegression

path_data = "data/data.xls"
path_out_txt = "out/out.txt"
path_out_png = "out/out.png"

data = xlrd.open_workbook(path_data)
table = data.sheet_by_index(0)
data_num = table.nrows - 1
pm25 = table.col_values(2)[1:]
pm25 = np.array(pm25)
temperature = table.col_values(8)[1:]
temperature = np.array(temperature)
wind = table.col_values(9)[1:]
wind = np.array(wind)
moisture = table.col_values(11)[1:]
moisture = np.array(moisture)
weather = table.col_values(10)[1:]
weather = np.array(weather)

use_min_max_scaler = True
use_all_data = True
step = 1
train_deep = 50
train_start = 50
predict_start = 51

print_log = True

assert step > 0
assert train_deep >= step and train_start >= train_deep
assert predict_start > train_start

regressor = SupervisedDBNRegression(hidden_layers_structure=[100],
                                    learning_rate_rbm=0.01,
                                    learning_rate=0.01,
                                    n_epochs_rbm=20,
                                    n_iter_backprop=200,
                                    batch_size=16,
                                    activation_function='relu')
min_max_scaler = MinMaxScaler()
open(path_out_txt, 'w').close()

if use_all_data:
    Data = np.concatenate((temperature[0:step], wind[0:step], weather[0:step], moisture[0:step]), axis=0)
else:
    Data = pm25[0:step]

Target = pm25[step]
predict = []
correct = []
loss_total = 0.0

for i in range(step + 1, data_num):

    if use_all_data:
        train_data_last = np.concatenate(
            (temperature[i - step:i], wind[i - step:i], weather[i - step:i], moisture[i - step:i]),
            axis=0)
    else:
        train_data_last = pm25[i - step:i]
    if print_log:
        print("train_data_last:", train_data_last.shape)
    Data = np.row_stack((Data, train_data_last))
    Target = np.row_stack((Target, pm25[i]))

    # predicting
    if i > predict_start:
        if use_min_max_scaler:
            train_data_last = train_data_last.reshape((1, train_data_last.size))
            tmp_test = min_max_scaler.transform(train_data_last)
        else:
            tmp_test = train_data_last

        tmp_pred = regressor.predict(tmp_test)
        with open(path_out_txt, 'a') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
            loss = math.sqrt(math.pow(tmp_pred[0][0] - pm25[i], 2)) / pm25[i]
            loss_total += loss
            f.write("%f\t%f\t%f\t%f\n" % (tmp_pred[0][0], pm25[i], loss, loss_total / (i - predict_start)))
        predict.append(tmp_pred[0][0])
        correct.append(pm25[i])
        plt.plot(range(i - predict_start), predict, 'r-o', range(i - predict_start), correct, 'b-o')
        plt.savefig(path_out_png)

    # training
    if i > train_start:
        if use_min_max_scaler:
            tmp_data = min_max_scaler.fit_transform(Data[i - train_deep:i])
        else:
            tmp_data = Data[i - train_deep:i]
        if print_log:
            print("Data:", Data.shape)
            print("tmp_data:", tmp_data.shape)
        regressor.fit(tmp_data, Target[i - train_deep:i, 0])

if print_log:
    print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(correct, predict), mean_squared_error(correct, predict)))
