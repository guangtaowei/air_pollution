import numpy as np
import xlrd
import math
import matplotlib.pyplot as plt

import os
import sys

# path_data = "data/data.xls"
path_data = "data/airdata.xlsx"

path_feather_png = "out/feather.png"

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

# 1.线图
#调用plt。plot来画图,横轴纵轴两个参数即可
#plt.figure(figsize=(28, 19))颜色不够用而且太密集看不清
plt.plot(hour, marker='o', label="hour",linewidth=0.3,markersize=1)
plt.plot(pm25, marker='o', label="pm2.5",linewidth=0.3,markersize=1)
plt.plot(o3, marker='o', label="o3",linewidth=0.3,markersize=1)
plt.plot(pm10, marker='o', label="pm10",linewidth=0.3,markersize=1)
plt.plot(so2, marker='o', label="so2",linewidth=0.3,markersize=1)
plt.plot(no2, marker='o', label="no2",linewidth=0.3,markersize=1)
plt.plot(co, marker='o', label="co",linewidth=0.3,markersize=1)
plt.plot(temperature, marker='o', label="temperature",linewidth=0.3,markersize=1)
plt.plot(wind, marker='o', label="wind",linewidth=0.3,markersize=1)
plt.plot(weather, marker='o', label="weather",linewidth=0.3,markersize=1)
plt.plot(moisture, marker='o', label="moisture",linewidth=0.3,markersize=1)
plt.plot(pressure, marker='o', label="pressure",linewidth=0.3,markersize=1)
plt.plot(precipitation, marker='o', label="precipitation",linewidth=0.3,markersize=1)
plt.legend(loc='best')
# python要用show展现出来图
#plt.show()
plt.savefig(path_feather_png)