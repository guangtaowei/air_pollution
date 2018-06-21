import os
import sys
import tensorflow as tf
import numpy as np

path_DBN = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"), "deep-belief-network")
sys.path.append(path_DBN)
from dbn.tensorflow import SupervisedDBNRegression

x_train = np.array(
    [[0, 22, 48, 26, 3, 25, 0.7, 4.7, 2, 60, 92, 1017.8, 0], [1, 13, 54, 26, 3, 19, 0.7, 4.4, 2, 10, 95, 1019.2, 2.6],
     [2, 9, 53, 20, 3, 20, 0.7, 4.7, 2, 10, 95, 1018.9, 1.2], [3, 14, 62, 20, 3, 15, 0.6, 4.9, 2, 60, 95, 1018.2, 1.3],
     [4, 18, 66, 25, 3, 14, 0.6, 5.3, 2, 0, 93, 1017.7, 0.7], [5, 16, 61, 28, 5, 24, 0.6, 5.3, 2, 10, 95, 1017.9, 0.6],
     [6, 15, 61, 24, 6, 24, 0.6, 5.6, 2, 60, 95, 1017, 1.1], [7, 21, 42, 29, 8, 41, 0.6, 6, 2, 0, 92, 1016.7, 0],
     [8, 18, 25, 32, 8, 56, 0.6, 6.2, 2, 0, 90, 1016.8, 0], [9, 23, 37, 32, 6, 46, 0.7, 5.8, 2, 60, 92, 1017.8, 0]])
y_train = np.array([[13], [9], [14], [18], [16], [15], [21], [18], [23], [22]])

x_test = np.array([[10, 22, 34, 34, 4, 48, 0.7, 5.5, 2, 10, 91, 1018.2, 0]])

regressor_DBN = SupervisedDBNRegression(hidden_layers_structure=[20,10,1],
                                        learning_rate_rbm=0.01,
                                        learning_rate=0.01,
                                        n_epochs_rbm=20,
                                        n_iter_backprop=200,
                                        batch_size=1,
                                        activation_function='sigmoid',
                                        verbose=False)

for i in range(9):
    print("\ni=", i)
    #print(x_train[i],y_train[i],x_train[i].shape)
    regressor_DBN.fit(x_train[i].reshape(1,x_train[i].shape[0]), y_train[i])
    pred = regressor_DBN.predict(x_train[i+1])
    print("\t", pred, y_train[i+1])
