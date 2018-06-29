import numpy as np
from sklearn.model_selection import train_test_split
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
import pandas as pd
from sklearn import svm
# from queue import Queue
from threading import Thread
from multiprocessing import Process, Queue
import os
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 默认为0：输出所有log信息
# 设置为1：进一步屏蔽INFO信息
# 设置为2：进一步屏蔽WARNING信息
# 设置为3：进一步屏蔽ERROR信息

np.random.seed(1337)  # for reproducibility
logging.basicConfig(level=logging.INFO)

# path_DBN = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"), "deep-belief-network")
path_DBN = os.path.join(os.path.join(os.getcwd( ), "models"), "deep-belief-network")
sys.path.append(path_DBN)
from dbn.tensorflow import SupervisedDBNRegression


def train_model(regressor_DBN, x_train, y_train, x_test):
    regressor_DBN.fit(x_train, y_train)
    pred = regressor_DBN.predict(x_test)
    return pred


def train_model_func(regressor_DBN, feature, label, path_out_png, path_out_txt, train_deep, regressor_DBN_tradition,start):
    print("Training model...")

    corrects = []
    predictions_trandition = []
    predictions_online = []
    RMSEs_trandition = []
    RMSEs_online = []

    for i in range(train_deep+start, label.shape[0]):
        '''x_train_trandition = np.array(feature[0:tradition_train_deep])
        y_trian_trandition = np.array(label[0:tradition_train_deep])'''
        x_train_online = np.array(feature[i - train_deep-start:i])
        y_trian_online = np.array(label[i - train_deep-start:i])
        x_test = np.array(feature[i])
        y_test = np.array(label[i:i + 1])[0]

        '''prediction_trandition = \
            train_model(regressor_DBN=regressor_DBN, x_train=x_train_trandition, y_train=y_trian_trandition,
                        x_test=x_test)[
                0][0]'''
        prediction_trandition = regressor_DBN_tradition.predict(x_test)[0][0]

        '''prediction_online = \
            train_model(regressor_DBN=regressor_DBN, x_train=x_train_online, y_train=y_trian_online, x_test=x_test)[0][
                0]'''
        if (i - train_deep-start) % 10 == 0:
            regressor_DBN.fit(x_train_online, y_trian_online)
        prediction_online = regressor_DBN.predict(x_test)[0][0]

        corrects.append(y_test)
        predictions_trandition.append(prediction_trandition)
        predictions_online.append(prediction_online)

        RMSE_trandition = math.sqrt(mean_squared_error(corrects, predictions_trandition))
        RMSE_online = math.sqrt(mean_squared_error(corrects, predictions_online))

        RMSEs_trandition.append(RMSE_trandition)
        RMSEs_online.append(RMSE_online)

        with open(path_out_txt, 'a') as f:
            f.write("pred_trad:%.2f\tpred_online:%.2f\tcorrect:%.2f\t\tRMSE_pred:%.15f\tRMSE_online:%.15f" % (
                prediction_trandition, prediction_online, y_test, RMSE_trandition, RMSE_online,))
            '''f.write("pred_online:%.2f\tcorrect:%.2f\t\tRMSE_online:%.15f" % (prediction_online, y_test, RMSE_online,))'''

        print("\t\ti=%d\t\tpred_trad:%.2f\tpred_online:%.2f\tcorrect:%.2f\t\tRMSE_pred:%.15f\tRMSE_online:%.15f" % (
            i, prediction_trandition, prediction_online, y_test, RMSE_trandition, RMSE_online,))
        '''print("\t\ti=%d\t\tpred_online:%.2f\tcorrect:%.2f\t\tRMSE_online:%.15f" % (
            i, prediction_online, y_test, RMSE_online,))'''

        x_range = range(i - train_deep-start + 1)
        plt.clf()
        plt.ylabel('value')
        plt.xlabel('hours')
        plt.title("DBN_online")
        plt.plot(x_range, predictions_trandition, marker='o', label="DBN_tradition")
        plt.plot(x_range, corrects, marker='o', label="correct")
        plt.plot(x_range, predictions_online, marker='o', label="DBN_online")
        plt.legend(loc='best')
        plt.savefig(path_out_png)

    print('Done.\nDBN_trad:\tR-squared: %f\nMSE: %f' % (
        r2_score(corrects, predictions_trandition), mean_squared_error(corrects, predictions_trandition)))
    print('DBN_online:\tR-squared: %f\nMSE: %f' % (
        r2_score(corrects, predictions_online), mean_squared_error(corrects, predictions_online)))


def main(data, target, path_out_png, path_out_txt, learning_rate_rbm=0.001, learning_rate=0.001, batch_size=2,
         train_deep=1, step=10, tradition_train_deep=10):
    feature = np.array([])
    for start in range(step, data.shape[0] + 1):
        feature = np.append(feature, data[start - step:start].values)
    label = target[step - 1:]
    feature = feature.reshape(label.shape[0], math.floor(feature.size / label.shape[0]))

    regressor_DBN = SupervisedDBNRegression(hidden_layers_structure=[20, 10, 2],
                                            learning_rate_rbm=learning_rate_rbm,
                                            learning_rate=learning_rate,
                                            n_epochs_rbm=200,
                                            n_iter_backprop=200,
                                            batch_size=batch_size,
                                            activation_function='sigmoid',
                                            verbose=False)

    x_train_trandition = np.array(feature[0:tradition_train_deep])
    y_trian_trandition = np.array(label[0:tradition_train_deep])
    regressor_DBN_tradition = SupervisedDBNRegression(hidden_layers_structure=[20, 10, 2],
                                                      learning_rate_rbm=learning_rate_rbm,
                                                      learning_rate=learning_rate,
                                                      n_epochs_rbm=200,
                                                      n_iter_backprop=200,
                                                      batch_size=batch_size,
                                                      activation_function='sigmoid',
                                                      verbose=False)
    regressor_DBN_tradition.fit(x_train_trandition, y_trian_trandition)

    train_model_func(regressor_DBN=regressor_DBN, feature=feature, label=label, path_out_png=path_out_png,
                     path_out_txt=path_out_txt, train_deep=train_deep, regressor_DBN_tradition=regressor_DBN_tradition,start=tradition_train_deep)


if __name__ == "__main__":
    path_out_png = "out/out.png"
    path_out_txt = "out/out.txt"
    open(path_out_txt, 'w').close()

    path_data = "data/airdata.csv"
    data = pd.read_csv(path_data, sep=",")

    target = data["pm25"]
    target = target.drop([0])

    data = data.drop([data.shape[0] - 1])
    data = data.drop(["date"], axis=1)

    sys.exit(main(data=data, target=target, path_out_png=path_out_png, path_out_txt=path_out_txt))
