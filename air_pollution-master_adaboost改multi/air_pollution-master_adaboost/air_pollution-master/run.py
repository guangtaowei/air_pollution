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

path_DBN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep-belief-network")
sys.path.append(path_DBN)
from dbn.tensorflow import SupervisedDBNRegression

# path_CRBM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CRBM-DBN")
# sys.path.append(path_CRBM)

# path_KCCA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PyKCCA")
# sys.path.append(path_KCCA)
# from kcca import *

from sklearn import svm
#from sklearn import PCA
logging.basicConfig(level=logging.INFO)

# path_data = "data/data.xls"
path_data = "data/airdata.xlsx"
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

assert step > 0
assert train_deep >= step and train_start >= train_deep
assert predict_start > train_start

regressor_DBN = SupervisedDBNRegression(hidden_layers_structure=[100],
                                        learning_rate_rbm=0.01,
                                        learning_rate=0.01,
                                        n_epochs_rbm=20,
                                        n_iter_backprop=200,
                                        batch_size=64,
                                        activation_function='sigmoid',
                                        verbose=False)
# regressor_AdaBoost = AdaBoostRegressor()
regressor_DBNAdaBoost = AdaBoostRegressor(SupervisedDBNRegression(hidden_layers_structure=[100],
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

regressor_MultiDBN = SupervisedDBNRegression(hidden_layers_structure=[100],
                                        learning_rate_rbm=0.01,
                                        learning_rate=0.01,
                                        n_epochs_rbm=20,
                                        n_iter_backprop=200,
                                        batch_size=64,
                                        activation_function='sigmoid',
                                        verbose=False)
regressor_MultiDBNAdaBoost = AdaBoostRegressor(SupervisedDBNRegression(hidden_layers_structure=[100],
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


#regressor_SVM = svm.SVR()

min_max_scaler = MinMaxScaler()

open(path_out_txt, 'w').close()

#if use_all_data:
#    Data = np.concatenate((pm25[0:step], temperature[0:step], wind[0:step], weather[0:step], moisture[0:step]), axis=0)
#else:
#    Data = pm25[0:step]
Data1 = pm25[0:step]
Data2 = np.concatenate((pm25[0:step], temperature[0:step], wind[0:step], weather[0:step], moisture[0:step]), axis=0)

Target = pm25[step]

correct = []
predict_DBN = []
predict_DBNAdaBoost = []
#predict_SVM = []
predict_MultiDBN = []
predict_MultiDBNAdaBoost = []



loss_total_DBN = 0.0
loss_total_DBNAdaBoost = 0.0
loss_total_MultiDBN = 0.0
loss_total_MultiDBNAdaBoost = 0.0


#loss_total_SVM = 0.0

for i in range(step + 1, data_num):

#    if use_all_data:
#        train_data_last = np.concatenate(
#            (pm25[i - step:i], temperature[i - step:i], wind[i - step:i], weather[i - step:i], moisture[i - step:i]),
#            axis=0)
 #   else:
    train_data1_last = pm25[i - step:i]
    logging.debug("train_data1_last:%s", train_data1_last.shape)
    Data1 = np.row_stack((Data1, train_data1_last))


    train_data2_last = np.concatenate(
           (pm25[i - step:i], temperature[i - step:i], wind[i - step:i], weather[i - step:i], moisture[i - step:i]),
           axis=0)
    logging.debug("train_data2_last:%s", train_data2_last.shape)
    Data2 = np.row_stack((Data2, train_data2_last))


    Target = np.row_stack((Target, pm25[i]))


    # predicting
    if i > predict_start:
        if use_min_max_scaler:
            train_data1_last = train_data1_last.reshape((1, train_data1_last.size))
            tmp_test1 = min_max_scaler.transform(train_data1_last)
            train_data2_last = train_data2_last.reshape((1, train_data2_last.size))
            tmp_test2 = min_max_scaler.transform(train_data2_last)

        else:
            tmp_test1 = train_data1_last
            tmp_test2 = train_data2_last
        tmp_pred_DBN = regressor_DBN.predict(tmp_test1)[0][0]
        tmp_pred_DBNAdaBoost = regressor_DBNAdaBoost.predict(tmp_test1)[0][0][0][0]
        tmp_pred_MultiDBN = regressor_MultiDBN.predict(tmp_test2)[0][0]
        tmp_pred_MultiDBNAdaBoost = regressor_MultiDBNAdaBoost.predict(tmp_test2)[0][0][0][0]

        #        tmp_pred_SVM = regressor_SVM.predict(tmp_test)[0]
        logging.info("==========================================")
        logging.info("pred_DBN:%s", tmp_pred_DBN)
        logging.info("pred_DBNAdaBoost:%s", tmp_pred_DBNAdaBoost)

        #        logging.info("pred_SVM:%s", tmp_pred_SVM)
        logging.info("pred_MultiDBN:%s", tmp_pred_MultiDBN)
        logging.info("pred_MultiDBNAdaBoost:%s", tmp_pred_MultiDBNAdaBoost)

        logging.info("correct:%s", pm25[i])
        logging.info("==========================================")
        with open(path_out_txt, 'a') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
            loss_DBN = math.sqrt(math.pow(tmp_pred_DBN - pm25[i], 2)) / pm25[i]
            loss_total_DBN += loss_DBN
            loss_DBNAdaBoost = math.sqrt(math.pow(tmp_pred_DBNAdaBoost - pm25[i], 2)) / pm25[i]
            loss_total_DBNAdaBoost += loss_DBNAdaBoost
            loss_MultiDBN = math.sqrt(math.pow(tmp_pred_MultiDBN - pm25[i], 2)) / pm25[i]
            loss_total_MultiDBN += loss_MultiDBN
            loss_MultiDBNAdaBoost = math.sqrt(math.pow(tmp_pred_MultiDBNAdaBoost - pm25[i], 2)) / pm25[i]
            loss_total_MultiDBNAdaBoost += loss_MultiDBNAdaBoost

            #            loss_SVM = math.sqrt(math.pow(tmp_pred_SVM - pm25[i], 2)) / pm25[i]
            #            loss_total_SVM += loss_SVM
            f.write(
                "p_D:%f\tp_AD:%f\tp_MD:%f\tp_MAD:%f\tc:%f\t\tl_D:%f\tl_AD:%f\tl_MD:%f\tl_MAD:%f\t\tl_avg_D:%f\tl_avg_AD:%f\tl_avg_MD:%f\tl_avg_MAD:%f\n" % (
                    tmp_pred_DBN, tmp_pred_DBNAdaBoost, tmp_pred_MultiDBN, tmp_pred_MultiDBNAdaBoost, pm25[i], loss_DBN,
                    loss_DBNAdaBoost, loss_MultiDBN, loss_MultiDBNAdaBoost,
                    loss_total_DBN / (i - predict_start), loss_total_DBNAdaBoost / (i - predict_start),
                    loss_total_MultiDBN / (i - predict_start), loss_total_MultiDBNAdaBoost / (i - predict_start)))
        predict_DBN.append(tmp_pred_DBN)
        predict_DBNAdaBoost.append(tmp_pred_DBNAdaBoost)
        predict_MultiDBN.append(tmp_pred_MultiDBN)
        predict_MultiDBNAdaBoost.append(tmp_pred_MultiDBNAdaBoost)

        correct.append(pm25[i])
        x_range = range(i - predict_start)
        plt.clf()
        plt.plot(x_range, predict_DBN, marker='o', label="DBN")
        plt.plot(x_range, predict_DBNAdaBoost, marker='o', label="DBNAdaBoost")
        plt.plot(x_range, predict_MultiDBN, marker='o', label="MultiDBN")
        plt.plot(x_range,  predict_MultiDBNAdaBoost, marker='o', label="MultiDBNAdaBoost")
        plt.plot(x_range, correct, marker='o', label="correct")
        plt.legend(loc='best')
        plt.savefig(path_out_png)

    # training
    if i > train_start:
        if use_min_max_scaler:
            tmp_data1 = min_max_scaler.fit_transform(Data1[i - train_deep:i])
            tmp_data2 = min_max_scaler.fit_transform(Data2[i - train_deep:i])
        else:
            tmp_data1 = Data1[i - train_deep:i]
            tmp_data2 = Data2[i - train_deep:i]
        logging.debug("Data1:%s", Data1.shape)
        logging.debug("tmp_data1:%s", tmp_data1.shape)


        regressor_DBN.fit(tmp_data1, Target[i - train_deep:i, 0])
        regressor_DBNAdaBoost.fit(tmp_data1, Target[i - train_deep:i, 0])
        logging.debug("Data2:%s", Data2.shape)
        logging.debug("tmp_data2:%s", tmp_data2.shape)
        regressor_MultiDBN.fit(tmp_data2, Target[i - train_deep:i, 0])
        regressor_MultiDBNAdaBoost.fit(tmp_data2, Target[i - train_deep:i, 0])

logging.info(
    'Done.\nDBN:\tR-squared: %f\nMSE: %f' % (r2_score(correct, predict_DBN), mean_squared_error(correct, predict_DBN)))
logging.info('DBNAdaBoost:\tR-squared: %f\nMSE: %f' % (
    r2_score(correct, predict_DBNAdaBoost), mean_squared_error(correct, predict_DBNAdaBoost)))
logging.info(
    'Done.\nMultiDBN:\tR-squared: %f\nMSE: %f' % (r2_score(correct, predict_MultiDBN), mean_squared_error(correct, predict_MultiDBN)))
logging.info('MultiDBNAdaBoost:\tR-squared: %f\nMSE: %f' % (
    r2_score(correct, predict_MultiDBNAdaBoost), mean_squared_error(correct, predict_MultiDBNAdaBoost)))
