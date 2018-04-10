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


def train_model(learning_rate, periods, batch_size, feature, label, path_out_png):
    path_DBN = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"), "deep-belief-network")
    sys.path.append(path_DBN)
    from dbn.tensorflow import SupervisedDBNRegression

    X_train, X_test, Y_train, Y_test = train_test_split(feature, label, test_size=0.2, shuffle=False)

    train_steps_per_period = X_train.shape[0] // periods  # floor
    test_steps_per_period = X_test.shape[0] // periods
    # print(X_train)
    # print(Y_train)

    '''regressor_DBN = SupervisedDBNRegression(hidden_layers_structure=[100],
                                            learning_rate_rbm=0.01,
                                            learning_rate=learning_rate,
                                            n_epochs_rbm=20,
                                            n_iter_backprop=200,
                                            batch_size=batch_size,
                                            activation_function='sigmoid',
                                            verbose=False)'''
    regressor_DBN = SupervisedDBNRegression(learning_rate=learning_rate, batch_size=batch_size, verbose=False)

    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range(0, periods):
        x_train = np.array(X_train[period * train_steps_per_period:(period + 1) * train_steps_per_period])
        # x_train=x_train.reshape(x_train.size,1)
        y_trian = np.array(Y_train[period * train_steps_per_period:(period + 1) * train_steps_per_period])
        # y_trian=y_trian.reshape(y_trian.size,1,1)
        # print(x_train)
        # print(y_trian)
        regressor_DBN.fit(x_train, y_trian)

        x_test = X_test[period * test_steps_per_period:(period + 1) * test_steps_per_period]
        y_test = Y_test[period * test_steps_per_period:(period + 1) * test_steps_per_period]
        predictions = regressor_DBN.predict(x_test)
        # predictions = np.array([predictions])
        # print(predictions.shape)
        # print(y_test.shape)

        root_mean_squared_error = math.sqrt(mean_squared_error(y_test, predictions))

        print("  period %02d : %0.2f" % (period, root_mean_squared_error))

        root_mean_squared_errors.append(root_mean_squared_error)

    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)
    plt.savefig(path_out_png)

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)


path_data = "data/airdata.csv"
# path_out_txt = "out/out_test.txt"
path_out_png = "out/out_test.png"

data = pd.read_csv(path_data, sep=",")

target = data["pm25"]
target = target.drop([0])

data = data.drop([data.shape[0] - 1])
data = data.drop(["date"], axis=1)
# print(data["date"])


learning_rate = 0.00001
batch_size = 1

#train_model(learning_rate=learning_rate, periods=10, batch_size=batch_size, feature=data, label=target,
 #           path_out_png=path_out_png)

print(data.size)
print(data)
print(data.shape)
print(math.floor(data.shape[0]/2))
