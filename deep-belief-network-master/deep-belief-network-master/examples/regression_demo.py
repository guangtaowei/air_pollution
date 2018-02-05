import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import xlrd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dbn.tensorflow import SupervisedDBNRegression


#from dbn. import SupervisedDBNClassification

data = xlrd.open_workbook('data.xls')
table = data.sheet_by_index(0)
pm25=table.col_values(2)[1:]
pm25=np.array(pm25)
temperature=table.col_values(8)[1:]
temperature=np.array(temperature)
wind=table.col_values(9)[1:]
wind=np.array(wind)
moisture=table.col_values(11)[1:]
moisture=np.array(moisture)
weather=table.col_values(10)[1:]
weather=np.array(weather)

Data=np.concatenate((pm25,temperature,wind,weather,moisture),axis=0)
Data=Data.reshape((5,738))

print (Data)
'''
# Loading dataset
boston = load_boston()
X, Y = boston.data, boston.target

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337)

# Data scaling
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)

# Training
regressor = SupervisedDBNRegression(hidden_layers_structure=[100],
                                    learning_rate_rbm=0.01,
                                    learning_rate=0.01,
                                    n_epochs_rbm=20,
                                    n_iter_backprop=200,
                                    batch_size=16,
                                    activation_function='relu')
regressor.fit(X_train, Y_train)

# Test
X_test = min_max_scaler.transform(X_test)
Y_pred = regressor.predict(X_test)
print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))
'''