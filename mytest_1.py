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

np.random.seed(1337)  # for reproducibility
logging.basicConfig(level=logging.INFO)


def train_model(learning_rate_rbm, learning_rate, batch_size, x_train, y_trian, x_test):
    path_DBN = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"), "deep-belief-network")
    sys.path.append(path_DBN)
    from dbn.tensorflow import SupervisedDBNRegression

    regressor_DBN = SupervisedDBNRegression(learning_rate_rbm=learning_rate_rbm, learning_rate=learning_rate,
                                            batch_size=batch_size, verbose=False)
    regressor_DBN.fit(x_train, y_trian)
    pred = regressor_DBN.predict(x_test)
    return pred


def train_model_func(learning_rate_rbm, learning_rate, batch_size, feature, label, path_out_png, pred_num, train_deep):
    X_train, X_test, Y_train, Y_test = train_test_split(feature, label, test_size=0.2, shuffle=False)

    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for deep in range(1, train_deep + 1):
        RMSE_total = 0
        for i in range(0, pred_num):
            x_train = np.array(X_train[X_train.shape[0] - i - deep:X_train.shape[0] - i])
            y_trian = np.array(Y_train[Y_train.shape[0] - i - deep:Y_train.shape[0] - i])
            x_test = np.array(X_test)
            y_test = np.array(Y_test)

            predictions = train_model(learning_rate_rbm=learning_rate_rbm, learning_rate=learning_rate,
                                      batch_size=batch_size, x_train=x_train,
                                      y_trian=y_trian, x_test=x_test)

            root_mean_squared_error = math.sqrt(mean_squared_error(y_test, predictions))
            print("\t\ti:\t", root_mean_squared_error)
            RMSE_total += root_mean_squared_error

        RMSE_avg = RMSE_total / pred_num
        root_mean_squared_errors.append(RMSE_avg)
        print("train_deep:", deep, "\tRMSE_avg:", RMSE_avg)

        # Output a graph of loss metrics over periods.
        # plt.subplot(1, 2, 2)
        plt.ylabel('RMSE')
        plt.xlabel('train_deep')
        plt.title("Root Mean Squared Error vs. Train Deep")
        plt.tight_layout()
        plt.plot(root_mean_squared_errors)
        plt.savefig(path_out_png)

    print("finished.")


path_data = "data/airdata.csv"
path_out_png = "out/out_test_TrianDeep.png"

data = pd.read_csv(path_data, sep=",")

target = data["pm25"]
target = target.drop([0])

data = data.drop([data.shape[0] - 1])
data = data.drop(["date"], axis=1)
# print(data["date"])


learning_rate_rbm = 0.01
learning_rate = 0.00001
batch_size = 1
pred_num = 3
train_deep = 200

train_model_func(learning_rate_rbm=learning_rate_rbm, learning_rate=learning_rate, batch_size=batch_size, feature=data,
                 label=target, path_out_png=path_out_png, pred_num=pred_num, train_deep=train_deep)
