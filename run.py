import numpy as np

np.random.seed(1337)  # for reproducibility
# from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor
# import GridSearchCV

import xlrd
import math
import matplotlib.pyplot as plt
import logging

import os
import sys

path_DBN = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"), "deep-belief-network")
sys.path.append(path_DBN)
from dbn.tensorflow import SupervisedDBNRegression

from sklearn import svm


def dada_handle(items, _use_all_data=True, _use_CCA_data=False, _use_pm25_history=True,
                _use_drop_least_importance=False, _start=0, _end=1):
    assert not (_use_all_data and _use_CCA_data)
    assert _use_all_data or _use_CCA_data or _use_drop_least_importance or _use_pm25_history
    assert (not _use_drop_least_importance) or (
            (not _use_all_data) and (not _use_CCA_data) and _use_drop_least_importance)

    if not use_pm25_history:
        items.pop('pm25')
    if use_drop_least_importance:
        items.pop('weather')
        items.pop('wind')
    elif use_CCA_data:  # pm10,co,temperature,moisture
        items.pop('hour')
        items.pop('o3')
        items.pop('so2')
        items.pop('no2')
        items.pop('wind')
        items.pop('weather')
        items.pop('pressure')
        items.pop('precipitation')

    re = []
    for item in items.values():
        re.append(item[_start:_end])
    re = np.array(re)
    re = re.reshape(re.size)
    return re


logging.basicConfig(level=logging.INFO)

# path_data = "data/data.xls"
path_data = "data/airdata.xlsx"
path_out_txt = "out/out.txt"
path_out_png = "out/out.png"

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

data.release_resources()
del data

use_min_max_scaler = True
use_all_data = True
use_CCA_data = False
use_pm25_history = True
use_drop_least_importance = False
use_deep = False
step = 1
train_deep = 120
train_start = 121
predict_start = 122

assert step > 0
assert train_deep >= step and train_start >= train_deep
assert predict_start > train_start
'''assert not (use_all_data and use_CCA_data)
assert (not use_pm25_history) or (use_all_data and use_pm25_history) or (use_CCA_data and use_pm25_history) or (
        use_drop_least_importance and use_pm25_history)
assert (not use_drop_least_importance) or ((not use_all_data) and (not use_CCA_data) and use_drop_least_importance)'''

regressor_DBN = SupervisedDBNRegression(hidden_layers_structure=[100],
                                        learning_rate_rbm=0.01,
                                        learning_rate=0.01,
                                        n_epochs_rbm=20,
                                        n_iter_backprop=200,
                                        batch_size=16,
                                        activation_function='sigmoid',
                                        verbose=False)
# regressor_AdaBoost = AdaBoostRegressor()
regressor_AdaBoost = AdaBoostRegressor(SupervisedDBNRegression(hidden_layers_structure=[100],
                                                               learning_rate_rbm=0.01,
                                                               learning_rate=0.01,
                                                               n_epochs_rbm=20,
                                                               n_iter_backprop=200,
                                                               batch_size=16,
                                                               activation_function='sigmoid',
                                                               verbose=False),
                                       loss="square",
                                       n_estimators=250,
                                       learning_rate=50)
regressor_SVM = svm.SVR()

min_max_scaler = MinMaxScaler()

open(path_out_txt, 'w').close()

'''if use_all_data:
    if use_pm25_history:
        Data = np.concatenate((hour[0:step], pm25[0:step], o3[0:step], pm10[0:step], so2[0:step], no2[0:step],
                               co[0:step], temperature[0:step], wind[0:step], weather[0:step], moisture[0:step],
                               pressure[0:step], precipitation[0:step]), axis=0)
    else:
        Data = np.concatenate((hour[0:step], o3[0:step], pm10[0:step], so2[0:step], no2[0:step], co[0:step],
                               temperature[0:step], wind[0:step], weather[0:step], moisture[0:step], pressure[0:step],
                               precipitation[0:step]), axis=0)
elif use_CCA_data:
    if use_pm25_history:
        Data = np.concatenate(
            (pm25[0:step], pm10[0:step], co[0:step], temperature[0:step], moisture[0:step], pressure[0:step]),
            axis=0)
    else:
        Data = np.concatenate((pm10[0:step], co[0:step], temperature[0:step], moisture[0:step], pressure[0:step]),
                              axis=0)
elif use_drop_least_importance:
    if use_pm25_history:
        Data = np.concatenate((hour[0:step], pm25[0:step], o3[0:step], pm10[0:step], so2[0:step], no2[0:step],
                               co[0:step], temperature[0:step], moisture[0:step], pressure[0:step],
                               precipitation[0:step]), axis=0)
    else:
        Data = np.concatenate((hour[0:step], o3[0:step], pm10[0:step], so2[0:step], no2[0:step], co[0:step],
                               temperature[0:step], moisture[0:step], pressure[0:step], precipitation[0:step]), axis=0)
else:
    Data = pm25[0:step]'''
Data = dada_handle(
    {"hour": hour, "pm25": pm25, "o3": o3, "pm10": pm10, "so2": so2, "no2": no2, "co": co, "temperature": temperature,
     "wind": wind, "weather": weather, "moisture": moisture, "pressure": pressure, "precipitation": precipitation},
    _use_all_data=use_all_data, _use_CCA_data=use_CCA_data, _use_pm25_history=use_pm25_history,
    _use_drop_least_importance=use_drop_least_importance, _start=0, _end=step)

Target = pm25[step]

correct = []
predict_DBN = []
predict_AdaBoost = []
predict_SVM = []
loss_total_DBN = 0.0
loss_total_AdaBoost = 0.0
loss_total_SVM = 0.0

logging.debug("data_num:%s", data_num)

for i in range(step + 1, data_num):

    '''if use_all_data:
        if use_pm25_history:
            train_data_last = np.concatenate((hour[i - step:i], pm25[i - step:i], o3[i - step:i], pm10[i - step:i],
                                              so2[i - step:i], no2[i - step:i], co[i - step:i],
                                              temperature[i - step:i], wind[i - step:i], weather[i - step:i],
                                              moisture[i - step:i], pressure[i - step:i], precipitation[i - step:i]),
                                             axis=0)
        else:
            train_data_last = np.concatenate((hour[i - step:i], o3[i - step:i], pm10[i - step:i], so2[i - step:i],
                                              no2[i - step:i], co[i - step:i], temperature[i - step:i],
                                              wind[i - step:i], weather[i - step:i], moisture[i - step:i],
                                              pressure[i - step:i], precipitation[i - step:i]), axis=0)
    elif use_CCA_data:
        if use_pm25_history:
            train_data_last = np.concatenate(
                (pm25[i - step:i], pm10[i - step:i], co[i - step:i], temperature[i - step:i], moisture[i - step:i],
                 pressure[i - step:i]),
                axis=0)
        else:
            train_data_last = np.concatenate(
                (pm10[i - step:i], co[i - step:i], temperature[i - step:i], moisture[i - step:i],
                 pressure[i - step:i]), axis=0)
    elif use_drop_least_importance:
        if use_pm25_history:
            train_data_last = np.concatenate((hour[i - step:i], pm25[i - step:i], o3[i - step:i], pm10[i - step:i],
                                              so2[i - step:i], no2[i - step:i], co[i - step:i], temperature[i - step:i],
                                              moisture[i - step:i], pressure[i - step:i], precipitation[i - step:i]),
                                             axis=0)
        else:
            train_data_last = np.concatenate((hour[i - step:i], o3[i - step:i], pm10[i - step:i], so2[i - step:i],
                                              no2[i - step:i], co[i - step:i], temperature[i - step:i],
                                              moisture[i - step:i], pressure[i - step:i], precipitation[i - step:i]),
                                             axis=0)
    else:
        train_data_last = pm25[i - step:i]'''
    train_data_last = dada_handle({"hour": hour, "pm25": pm25, "o3": o3, "pm10": pm10, "so2": so2, "no2": no2, "co": co,
                                   "temperature": temperature, "wind": wind, "weather": weather, "moisture": moisture,
                                   "pressure": pressure, "precipitation": precipitation}, _use_all_data=use_all_data,
                                  _use_CCA_data=use_CCA_data, _use_pm25_history=use_pm25_history,
                                  _use_drop_least_importance=use_drop_least_importance, _start=i - step, _end=i)

    logging.debug("train_data_last:%s", train_data_last.shape)
    Data = np.row_stack((Data, train_data_last))
    Target = np.row_stack((Target, pm25[i]))

    # predicting
    if i > predict_start:
        if use_min_max_scaler:
            train_data_last = train_data_last.reshape((1, train_data_last.size))
            tmp_test = min_max_scaler.transform(train_data_last)
        else:
            train_data_last = train_data_last.reshape((1, train_data_last.size))
            tmp_test = train_data_last

        logging.debug("train_data_last:%s", train_data_last)
        logging.debug("tmp_test:%s", tmp_test)
        tmp_pred_DBN = regressor_DBN.predict(tmp_test)[0][0]
        tmp_pred_AdaBoost = regressor_AdaBoost.predict(tmp_test)[0][0][0][0]
        tmp_pred_SVM = regressor_SVM.predict(tmp_test)[0]
        logging.info("==========================================")
        logging.info("pred_DBN:%s", tmp_pred_DBN)
        logging.info("pred_AdaBoost:%s", tmp_pred_AdaBoost)
        logging.info("pred_SVM:%s", tmp_pred_SVM)
        logging.info("correct:%s", pm25[i])
        logging.info("==========================================")

        predict_DBN.append(tmp_pred_DBN)
        predict_AdaBoost.append(tmp_pred_AdaBoost)
        predict_SVM.append(tmp_pred_SVM)
        correct.append(pm25[i])

        with open(path_out_txt, 'a') as f:
            loss_DBN = math.sqrt(math.pow(tmp_pred_DBN - pm25[i], 2)) / pm25[i]
            loss_total_DBN += loss_DBN
            loss_AdaBoost = math.sqrt(math.pow(tmp_pred_AdaBoost - pm25[i], 2)) / pm25[i]
            loss_total_AdaBoost += loss_AdaBoost
            loss_SVM = math.sqrt(math.pow(tmp_pred_SVM - pm25[i], 2)) / pm25[i]
            loss_total_SVM += loss_SVM
            f.write(
                "p_D:%f\tp_A:%f\tp_S:%f\tc:%f\t\tl_D:%f\tl_A:%f\tl_S:%f\t\tR2_D:%f\tR2_A:%f\tR2_S:%f\t\tl_avg_D:%f\tl_avg_A:%f\tl_avg_S:%f\n" % (
                    tmp_pred_DBN, tmp_pred_AdaBoost, tmp_pred_SVM, pm25[i], loss_DBN, loss_AdaBoost, loss_SVM,
                    r2_score(correct, predict_DBN), r2_score(correct, predict_AdaBoost),
                    r2_score(correct, predict_SVM), loss_total_DBN / (i - predict_start),
                    loss_total_AdaBoost / (i - predict_start),
                    loss_total_SVM / (i - predict_start)))

        x_range = range(i - predict_start)
        plt.clf()
        plt.plot(x_range, predict_DBN, marker='o', label="DBN")
        plt.plot(x_range, predict_AdaBoost, marker='o', label="AdaBoost")
        plt.plot(x_range, predict_SVM, marker='o', label="SVM")
        plt.plot(x_range, correct, marker='o', label="correct")
        plt.legend(loc='best')
        plt.savefig(path_out_png)

    # training
    if i > train_start:
        if use_min_max_scaler:
            if use_deep:
                tmp_data = min_max_scaler.fit_transform(Data[i - train_deep:i])
            else:
                tmp_data = min_max_scaler.fit_transform(Data)
        else:
            if use_deep:
                tmp_data = Data[i - train_deep:i]
            else:
                tmp_data = Data
        logging.debug("Data:%s", Data.shape)
        logging.debug("tmp_data:%s", tmp_data.shape)
        if use_deep:
            regressor_DBN.fit(tmp_data, Target[i - train_deep:i, 0])
            regressor_AdaBoost.fit(tmp_data, Target[i - train_deep:i, 0])
            regressor_SVM.fit(tmp_data, Target[i - train_deep:i, 0])
        else:
            regressor_DBN.fit(tmp_data, Target[:, 0])
            regressor_AdaBoost.fit(tmp_data, Target[:, 0])
            regressor_SVM.fit(tmp_data, Target[:, 0])

logging.info(
    'Done.\nDBN:\tR-squared: %f\nMSE: %f' % (r2_score(correct, predict_DBN), mean_squared_error(correct, predict_DBN)))
logging.info('AdaBoost:\tR-squared: %f\nMSE: %f' % (
    r2_score(correct, predict_AdaBoost), mean_squared_error(correct, predict_AdaBoost)))
