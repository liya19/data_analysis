import pandas as pd
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask
import statsmodels.tsa.stattools as stattools
from statsmodels.tsa.seasonal import seasonal_decompose


# периодчески собирает данные, каждые 5 минут, вызывает 14 стр
def scavenge_data_periodically():
    # работает 1 раз, настраивается на момент времени
    while True:
        # старт идет с 00 или 05
        if datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').endswith(('0:00', '5:00')):
            scavenge_data()
            print('data scavenged')
            break
    timing = time.time()
    # бесконечно вызывает
    while True:
        if time.time() - timing > 300.0:
            timing = time.time()
            print("5 mins passed: data scavenged")
            scavenge_data()


def scavenge_data():
    data = pd.read_csv('final_data.csv', index_col=0)
    while True:
        try:
            gathered_data = pd.read_json('http://data.kzn.ru:8082/api/v0/dynamic_datasets/bus.json', orient='DataFrame')
        except Exception:
            print('Error getting data, retrying')
            continue
        else:
            gathered_data = gathered_data['data']
            # конвертация с сервера
            new_data = gathered_data.apply(lambda x: pd.DataFrame.from_dict(x, orient='index'))
            new_data = pd.concat(new_data.values, join='outer', axis=1).T.reset_index(drop=True)
            # новый список, при каждом выводе
            new_data.to_csv('lastData.csv')
            data = data.append(new_data, ignore_index=True)
            # всё
            data.to_csv('final_data.csv')
            break


def get_data():
    return pd.read_csv('final_data.csv', index_col=0, parse_dates=[5], dayfirst=True,
                       dtype={'GaragNumb': np.int, 'Graph': np.int, 'Smena': np.int, })


# округление времени
def process_data(df, round_param='5min'):
    df['TimeNav'] = df['TimeNav'].apply(lambda dt: dt.round(round_param))
    df.set_index(df['TimeNav'], inplace=True)
    return df


# группирирует timeNav и аггрериуется (берется средняя скорость)
def aggregate_data(df, how=lambda x: x.mean()):
    return df.groupby(['TimeNav']).agg(how)


# тест дики-фуллера
def adfuller(df):
    test = stattools.adfuller(df)
    print('adf: ', test[0])
    print('p-value: ', test[1])
    print('Critical values: ', test[4])
    if test[0] > test[4]['5%']:
        print('есть единичные корни, ряд не стационарен')
    else:
        print('единичных корней нет, ряд стационарен')


# scavenge_data_periodically()
dataset = get_data()
dataset = process_data(dataset, '5min')
speed_data = aggregate_data(dataset.Speed, lambda x: x.mean())
print(speed_data)

# Actual only
speed_data.plot(figsize=(18, 6), label='Actual values')

plt.legend(loc='upper left')
plt.savefig('graph.png')
plt.show()

# Actual + Rolling average + Rolling Std
speed_data.plot(figsize=(18, 6), label='Actual values')
rolling = speed_data.rolling(window=24 * 12).mean().plot(figsize=(18, 6), color='red', label='Rolling Average')
std = speed_data.rolling(window=24 * 12).std().plot(figsize=(18, 6), color='black', label='Rolling Std')
plt.legend(loc='best')
plt.show()

# ADFULLER
adfuller(speed_data)

# Time Series Components
decomposition = seasonal_decompose(x=speed_data, period=24 * 12)
plt.subplot(411)
plt.plot(speed_data, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

print(speed_data.describe())
