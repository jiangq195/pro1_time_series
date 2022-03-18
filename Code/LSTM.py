import os

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from pandas import read_csv
from pandas import DataFrame
from datetime import datetime
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from pandas import concat

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from numpy import concatenate

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

import math
import matplotlib.pyplot as plt
import matplotlib as mpl

from keras import Input, Model
from keras.layers import Dense
from keras.models import load_model

from Code.utils import data_split, data_split_LSTM, isolutionforest, plot_curve, RMSE, MAPE


def visualize(history):
    plt.rcParams['figure.figsize'] = (10.0, 6.0)
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def LSTM_Model(X_train, Y_train):
    os.makedirs('../checkpoints/LSTM', exist_ok=True)
    filepath = "../checkpoints/LSTM/LSTM_{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='auto', period=10)
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))  # 已经确定10步长
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=2, shuffle=True,
                        callbacks=[checkpoint])
    return model, history


if __name__ == '__main__':

    plt.rcParams['figure.figsize'] = (10.0, 5.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    dataset = pd.read_csv('../data/initial_0826.csv', header=0, index_col=0, parse_dates=True)
    data = dataset.values.reshape(-1)

    values = dataset.values
    # groups = [0, 1, 2, 3]
    # fig, axs = plt.subplots(1)

    df = pd.DataFrame(dataset)  # 整体数据的全部字典类型
    do = df['NH4_hlx']  # 返回溶解氧那一列，用字典的方式

    DO = []
    for i in range(0, len(do)):
        DO.append([do[i]])
    scaler_DO = MinMaxScaler(feature_range=(0, 1))
    DO = scaler_DO.fit_transform(DO)

    # 孤立森林
    DO = isolutionforest(DO)

    c = int(len(DO) * .85)
    lookback_window = 11

    # 数组划分为不同的数据集
    X1_train, Y1_train, X1_test, Y1_test = data_split(DO, c, lookback_window)  # TCN
    X2_train, Y2_train, X2_test, Y2_test = data_split_LSTM(X1_train, Y1_train, X1_test, Y1_test)

    train_mode = True
    if train_mode:
        # 训练模型
        model_DO_LSTM, his = LSTM_Model(X2_train, Y2_train)
        # visualize(his)
    else:
        # 记载已经保存的模型
        model_DO_LSTM = load_model("../checkpoints/LSTM/LSTM_100.h5")

    # 测试集拟合
    Y2_train_predict = model_DO_LSTM.predict(X2_train)
    # 变回原来的值,inverse_transform
    Y2_train_predict = scaler_DO.inverse_transform(Y2_train_predict)
    Y2_train_true = scaler_DO.inverse_transform(Y2_train)

    Y2_test_predict = model_DO_LSTM.predict(X2_test)

    Y2_test_true = scaler_DO.inverse_transform(Y2_test)
    Y2_test_predict = scaler_DO.inverse_transform(Y2_test_predict)

    rmse = format(RMSE(Y2_test_true, Y2_test_predict), '.4f')
    mape = format(MAPE(Y2_test_true, Y2_test_predict), '.4f')
    r2 = format(r2_score(Y2_test_true, Y2_test_predict), '.4f')
    mae = format(mean_absolute_error(Y2_test_true, Y2_test_predict), '.4f')
    plot_curve(Y2_test_true, Y2_test_predict, rmse, mae, mape, r2, 4, 'test', 'LSTM')

    print('RMSE:' + str(rmse) + '\n' + 'MAE:' + str(mae) + '\n' + 'MAPE:' + str(mape) + '\n' + 'R2:' + str(r2))
    eval_indicator = pd.DataFrame({'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}, index=['values'])
    eval_indicator.to_csv('../pictures/LSTM/eval_indicator.csv', index=False)
