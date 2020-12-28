import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import numpy
from pandas import datetime
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import matplotlib
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures

import plotly.graph_objs as go
import streamlit as st

url = "https://ncov2019.live/data/world"

r = requests.get(url)
df_list = pd.read_html(r.text)
world_df = df_list[2]
world_df = world_df[1:world_df.shape[0] - 1]

world_df['Tests'].replace("Unknown", 0, inplace=True)
world_df['Tests'] = pd.to_numeric(world_df['Tests'])
world_df['Active'].replace("Unknown", 0, inplace=True)
world_df['Active'] = pd.to_numeric(world_df['Active'])
world_df['Recovered'].replace("Unknown", 0, inplace=True)
world_df['Recovered'] = pd.to_numeric(world_df['Recovered'])
world_df['Confirmed'].replace("Unknown", 0, inplace=True)
world_df['Confirmed'] = pd.to_numeric(world_df['Confirmed'])
world_df['Deceased'].replace("Unknown", 0, inplace=True)
world_df['Deceased'] = pd.to_numeric(world_df['Deceased'])



def printFig(dataFrame, xType, yType):
    df = dataFrame.sort_values(by=yType, ascending=False).head(25)
    plt.bar(df[xType], df[yType])
    plt.xticks(df[xType], rotation='vertical')
    plt.yticks(df[yType])
    return plt


def printMap():
    import country_converter as coco
    Names = []
    for i in range(1, 215):
        Names.append(world_df.iloc[i]['Name'])

    standard_names = coco.convert(names=Names, to='ISO3')

    map_data = world_df[world_df['Name'] != 'TOTAL']
    print(len(Names))

    map_data = map_data[:214]
    map_data['code'] = standard_names
    map_data['code'] = map_data['code'].shift(1)

    choropleth_data = map_data[map_data['code'] != "NaN"]
    data = dict(
        type='choropleth',
        locations=choropleth_data['code'],
        z=choropleth_data['Confirmed'],
        text=choropleth_data['Confirmed'],
        marker=dict(line=dict(color='rgb(255,255,255)', width=2)),
        colorbar={'title': "Confirmed Cases"}
    )

    # Теперь создадим макет для графика
    layout = dict(
        title='World COVID-19 Stats',
        width=1080,
        height=900,
        geo=dict(
            showframe=False,
            projection={'type': 'mercator'}
        )
    )
    choromap = go.Figure(data=[data], layout=layout)
    return choromap


def printGroupOfProps(dataFrame, xType, yType1, yType2):
    df = dataFrame.sort_values(yType2, ascending=False)
    fig = go.Figure(data=[
        go.Bar(
            x=df[xType],
            y=df[yType1].head(20),
            name=yType1,
            marker_color="indianred"
        ),
        go.Bar(
            x=df[xType],
            y=df[yType2].head(20),
            name=yType2,
            marker_color="lightsalmon"
        )
    ])
    fig.update_layout(barmode='group')
    return fig


# получение последних данных по us
url = "https://ncov2019.live/data/unitedstates"

r = requests.get(url)
df_list = pd.read_html(r.text)
us_df = df_list[2]

confirmed_df = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

cols = confirmed_df.keys()
confirmed = confirmed_df.loc[:4, cols[4]:cols[-1]]

dates = confirmed.keys()

worldcases = []

for i in ((dates)):
    confirmed_sum = confirmed[i].sum()

    worldcases.append(confirmed_sum)

days_in_future = 10
future_forcast = np.array([i for i in range(len(dates) + days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]

start = '12/20/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

days_from_1_20 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_from_1_20[50:],
                                                                                            worldcases[50:],
                                                                                            test_size=0.15,
                                                                                            shuffle=False)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def SVM():
    html_string = "<h2>Support Vector Machine Model</h2>"
    st.markdown(html_string, unsafe_allow_html=True)
    svm_confirmed = SVR(shrinking=True, kernel='poly', gamma=0.01, epsilon=1, degree=3, C=0.1)
    svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
    svm_pred = svm_confirmed.predict(future_forcast)

    svm_test_pred = svm_confirmed.predict(X_test_confirmed)
    plt.figure(figsize=(20, 15))
    plt.plot(y_test_confirmed)
    plt.plot(svm_test_pred)
    plt.legend(['Test Data', 'SVM Predictions'])
    st.pyplot(plt)

    def MAE():
        return 'MAE: ' + str(mean_absolute_error(svm_test_pred, y_test_confirmed))

    def MSE():
        return 'MSE: ' + str(mean_squared_error(svm_test_pred, y_test_confirmed))

    st.markdown(MAE())
    st.markdown(MSE())

    st.markdown('Средняя абсолютная процентная погрешность SVM составляет' +
                str(mean_absolute_percentage_error(y_test_confirmed, svm_test_pred)))


confirmed_df = confirmed_df.T
confirmed_df = confirmed_df.rename(columns=confirmed_df.iloc[1])
confirmed_df = confirmed_df[4:]
confirmed_df['Total_cases'] = confirmed_df.sum(axis=1)
confirmed_df.reset_index(level=0, inplace=True)

confirmed_df['dates'] = pd.to_datetime(confirmed_df['index'])
time_series_analysis_df = confirmed_df[['Total_cases', 'dates']]
time_series_analysis_df = time_series_analysis_df.set_index('dates')


def sarimax():
    html_string = "<h2>ARIMA</h2>"
    st.markdown(html_string, unsafe_allow_html=True)
    decomposition = sm.tsa.seasonal_decompose(time_series_analysis_df, model='additive')
    fir = decomposition.plot()
    matplotlib.rcParams['figure.figsize'] = [20.0, 15.0]

    import itertools

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    mod = sm.tsa.statespace.SARIMAX(time_series_analysis_df,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    st.table(results.summary().tables[1])

    st.pyplot(plt)

    pred = results.get_prediction(start=pd.to_datetime('2020-09-01'), dynamic=False)
    pred_ci = pred.conf_int()
    ax = time_series_analysis_df.plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of cases')
    plt.legend()

    y_forecasted = pred.predicted_mean
    y_truth = time_series_analysis_df['Total_cases']['2020-09-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    st.text('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    st.text('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    st.markdown(mean_absolute_percentage_error(y_truth, y_forecasted))

    pred_uc = results.get_forecast(steps=100)
    pred_ci = pred_uc.conf_int()
    ax = time_series_analysis_df['Total_cases'].plot(label='Total_cases', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_title('COVID-19')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Cases')
    plt.legend()
    st.pyplot(plt)


todo_selectbox = st.sidebar.selectbox(
    "Что вы хотите сделать?",
    ("Ознакомиться с данными", "Обучение модели")
)


def PR():

    html_string = "<h2>Полиномиальная регрессия</h2>"
    st.markdown(html_string, unsafe_allow_html=True)
    # преобразовываем наши данные для полиномиальной регрессии
    poly = PolynomialFeatures(degree=5)
    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
    poly_future_forcast = poly.fit_transform(future_forcast)

    # полиномиальная регрессия
    linear_model = LinearRegression(normalize=True, fit_intercept=False)
    linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
    test_linear_pred = linear_model.predict(poly_X_test_confirmed)
    linear_pred = linear_model.predict(poly_future_forcast)
    st.markdown('MAE:' + str(mean_absolute_error(test_linear_pred, y_test_confirmed)))
    st.markdown('MSE:' + str(mean_squared_error(test_linear_pred, y_test_confirmed)))

    st.markdown('Средняя абсолютная процентная погрешность LR составляет ' +
                str(mean_absolute_percentage_error(y_test_confirmed, test_linear_pred)))

    plt.figure(figsize=(20, 15))
    plt.plot(y_test_confirmed)
    plt.plot(test_linear_pred)
    plt.legend(['Test Data', 'Polynomial Regression Predictions'])
    st.pyplot(plt)


def lstm_draw():

    html_string = "<h2>Нейронка LSTM</h2>"
    st.markdown(html_string, unsafe_allow_html=True)
    X = time_series_analysis_df.values
    train, test = X[0:230], X[230:]
    print(train.shape, test.shape)

    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        # make prediction
        predictions.append(history[-1])
        # observation
        history.append(test[i])
    # report performance
    rmse = sqrt(mean_squared_error(test, predictions))

    print('RMSE: %.3f' % rmse)
    print(mean_absolute_percentage_error(test, predictions))
    from matplotlib import pyplot

    # line plot of observed vs predicted
    pyplot.plot(test, label='Test')
    pyplot.plot(predictions, label='Predictions')

    path = r'C:/Users/Liya/AD/project'

    time_series_analysis_df.to_csv(path + '\series.csv')

    def parser(x):
        return datetime.datetime.strptime(x, '%Y-%m-%d')

    series = pd.read_csv('C:/Users/Liya/AD/project/series.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,
                         date_parser=parser)
    print(series.head())

    def timeseries_to_supervised(data, lag=1):
        df = pd.DataFrame(data)
        columns = [df.shift(i) for i in range(1, lag + 1)]
        columns.append(df)
        df = pd.concat(columns, axis=1)
        df.fillna(0, inplace=True)
        return df

    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return pd.Series(diff)

    def inverse_difference(history, yhat, interval=1):
        return yhat + history[-interval]

    def scale(train, test):
        # fit scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train)
        # transform train
        train = train.reshape(train.shape[0], train.shape[1])
        train_scaled = scaler.transform(train)
        # transform test
        test = test.reshape(test.shape[0], test.shape[1])
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled

    # inverse scaling for a forecasted value
    def invert_scale(scaler, X, value):
        new_row = [x for x in X] + [value]
        array = numpy.array(new_row)
        array = array.reshape(1, len(array))
        inverted = scaler.inverse_transform(array)
        return inverted[0, -1]

    # fit an LSTM network to training data
    def fit_lstm(train, batch_size, nb_epoch, neurons):
        X, y = train[:, 0:-1], train[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        for i in range(nb_epoch):
            model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
            model.reset_states()
        return model

    # make a one-step forecast
    def forecast_lstm(model, batch_size, X):
        X = X.reshape(1, 1, len(X))
        yhat = model.predict(X, batch_size=batch_size)
        return yhat[0, 0]

    series = pd.read_csv('C:/Users/Liya/AD/project/series.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,
                         date_parser=parser)
    print(series.head())

    raw_values = series.values
    diff_values = difference(raw_values, 1)

    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    # split data into train and test-sets
    train, test = supervised_values[0:230], supervised_values[230:]

    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)

    # fit the model
    lstm_model = fit_lstm(train_scaled, 1, 1500, 1)
    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)

    # walk-forward validation on the test data
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
        # store forecast
        predictions.append(yhat)
        expected = raw_values[len(train) + i + 1]
        print('day=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

    # report performance
    rmse = sqrt(mean_squared_error(raw_values[231:], predictions))
    st.markdown('Test RMSE: %.3f' + str(rmse))

    # line plot of observed vs predicted
    pyplot.plot(raw_values[231:])
    pyplot.plot(predictions)
    st.pyplot(pyplot)

    st.markdown('Средняя абсолютная процентная ошибка LSTM составляет' +
                str(mean_absolute_percentage_error(raw_values[231:], predictions)))


def TS():
    pass


if todo_selectbox == "Ознакомиться с данными":
    visualize_selectbox = st.sidebar.selectbox(
        "Отображение",
        ("Таблица", "Диаграмма", "Двойная диаграмма", "На карте")
    )
    if visualize_selectbox == "Таблица":

        st.table(world_df)

    elif visualize_selectbox == "Диаграмма":
        sort_selectbox = st.sidebar.selectbox(
            "Сортировать по",
            ("Confirmed", "Tests", "Recovered", "Deceased", "Active")
        )
        st.pyplot(printFig(world_df, 'Name', sort_selectbox))

    elif visualize_selectbox == "Двойная диаграмма":
        sort_selectbox1 = st.sidebar.selectbox(
            "Сортировать по",
            ("Confirmed", "Tests", "Recovered", "Deceased", "Active")
        )
        sort_selectbox2 = st.sidebar.selectbox(
            "Сортировать по",
            ("Confirmed", "Tests", "Recovered", "Deceased", "Active"), key=3
        )
        st.plotly_chart(printGroupOfProps(world_df, 'Name', sort_selectbox1, sort_selectbox2))

    elif visualize_selectbox == "На карте":
        st.plotly_chart(printMap())
elif todo_selectbox == "Обучение модели":
    visualize_selectbox = st.sidebar.selectbox(
        "Модель",
        ("SVM", "PR", "LSTM", "ARIMA")
    )

    if visualize_selectbox == "SVM":
        SVM()
    elif visualize_selectbox == "PR":
        PR()
    elif visualize_selectbox == "LSTM":
        lstm_draw()
    elif visualize_selectbox == "ARIMA":
        sarimax()
    elif visualize_selectbox == "ARIMA":
        pass
