import pandas as pd
import os
import matplotlib.pyplot as plt

data_path = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"), "airdata.csv")
data = pd.read_csv("data/airdata.csv", sep=",")
# print(data)
print(data.describe())
data.pop("气压")
#data.hist("pm25")
print(data.plot())
plt.savefig("out/dataframe.png")
