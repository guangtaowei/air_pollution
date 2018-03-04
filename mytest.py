import numpy as np
import xlrd


def mytest(items, _use_all_data=True, _use_CCA_data=False, _use_pm25_history=True, _use_drop_least_importance=False,
           _start=0, _end=1):
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


path_data = "data/airdata.xlsx"

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

use_all_data = True
use_CCA_data = False
use_pm25_history = True
use_drop_least_importance = False

step = 2

re_data = mytest(
    {"hour": hour, "pm25": pm25, "o3": o3, "pm10": pm10, "so2": so2, "no2": no2, "co": co, "temperature": temperature,
     "wind": wind, "weather": weather, "moisture": moisture, "pressure": pressure, "precipitation": precipitation},
    _use_all_data=use_all_data, _use_CCA_data=use_CCA_data, _use_pm25_history=use_pm25_history,
    _use_drop_least_importance=use_drop_least_importance, _start=0, _end=step)
print(re_data)
print(re_data.shape)

Data = np.concatenate((hour[0:step], pm25[0:step], o3[0:step], pm10[0:step], so2[0:step], no2[0:step],
                       co[0:step], temperature[0:step], wind[0:step], weather[0:step], moisture[0:step],
                       pressure[0:step], precipitation[0:step]), axis=0)
print(Data)
print(Data.shape)
