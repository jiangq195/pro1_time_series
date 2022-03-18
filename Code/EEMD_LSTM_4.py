import os

import numpy as np
import pandas as pd

from pandas import read_csv
from pandas import DataFrame
from datetime import datetime
from matplotlib import pyplot
from pylab import mpl

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest

from pandas import concat
from PyEMD import EEMD

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout
from keras.layers import Activation

import math
import matplotlib.pyplot as plt

from keras import Input, Model
from keras.layers import Dense
from keras.models import load_model
from Code.utils import plot_curve, data_split, isolutionforest, imf_data, data_split_LSTM, RMSE, MAPE, data_split_2, \
    visualize, data_split_test


def LSTM_Model(X_train, Y_train, model_name, i):
    os.makedirs(f'../checkpoints/{model_name}', exist_ok=True)
    filepath = f"../checkpoints/{model_name}/{model_name}_imf" + str(i) + "_{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='auto', period=10)
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))  # 已经确定10步长
    '''
    ,return_sequences = True
    如果设置return_sequences = True，该LSTM层会返回每一个time step的h，
    那么该层返回的就是1个由多个h组成的2维数组了，如果下一层不是可以接收2维数组
    的层，就会报错。所以一般LSTM层后面接LSTM层的话，设置return_sequences = True，
    如果接全连接层的话，设置return_sequences = False。
    '''
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=2, shuffle=True,
                        callbacks=[checkpoint])
    return model, history


if __name__ == '__main__':
    model_name = 'EEMD_LSTM_4'
    plt.rcParams['figure.figsize'] = (10.0, 5.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    dataset1 = pd.read_csv('../data/data_part1.csv', header=0, index_col=0, parse_dates=True)
    dataset2 = pd.read_csv('../data/data_part2.csv', header=0, index_col=0, parse_dates=True)
    dataset3 = pd.read_csv('../data/data_part3.csv', header=0, index_col=0, parse_dates=True)
    DO1 = dataset1['NH4_hlx'].values.reshape(-1, 1)
    DO2 = dataset2['NH4_hlx'].values.reshape(-1, 1)
    DO3 = dataset3['NH4_hlx'].values.reshape(-1, 1)
    # data = dataset.values.reshape(-1)
    #
    # values = dataset.values
    # groups = [0, 1, 2, 3]
    # # fig, axs = plt.subplots(1)
    #
    # df = pd.DataFrame(dataset)  # 整体数据的全部字典类型
    # do = df['NH4_hlx']  # 返回溶解氧那一列，用字典的方式
    #
    # DO = do.values.reshape(do.shape[0], 1)

    # DO = []
    # for i in range(0, len(do)):
    #     DO.append([do[i]])
    # tmp = DO

    # 孤立森林
    DO1 = isolutionforest(DO1)
    DO2 = isolutionforest(DO2)
    DO3 = isolutionforest(DO3)

    # # 归一化
    # scaler_DO = MinMaxScaler(feature_range=(0, 1))
    # DO = scaler_DO.fit_transform(DO)

    # 标准化
    scaler_DO1 = StandardScaler()
    DO1 = scaler_DO1.fit_transform(DO1)
    scaler_DO2 = StandardScaler()
    DO2 = scaler_DO1.fit_transform(DO2)
    scaler_DO3 = StandardScaler()
    DO3 = scaler_DO1.fit_transform(DO3)
    # scaler_DO2 = StandardScaler()
    # len2 = len(DO2)
    # DO23 = scaler_DO2.fit_transform(np.concatenate((DO2, DO3), axis=0))
    # DO2 = DO23[:len2]
    # DO3 = DO23[len2:]

    eemd1 = EEMD()
    eemd1.noise_seed(11111)
    imfs1 = eemd1.eemd(DO1.reshape(-1), None, -1)
    print('imfs1 length: ', len(imfs1))

    eemd2 = EEMD()
    eemd2.noise_seed(22222)
    imfs2 = eemd2.eemd(DO2.reshape(-1), None, 8)
    print('imfs2 length: ', len(imfs2))

    eemd3 = EEMD()
    eemd3.noise_seed(33333)
    imfs3 = eemd3.eemd(DO3.reshape(-1), None, 8)
    print('imfs3 length: ', len(imfs3))

    # eemd23 = EEMD()
    # imfs23 = eemd23.eemd(DO23.reshape(-1), None, -1)
    # print('imfs length: ', len(imfs23))

    lookback_window = 11
    imfs_prediction = []

    train_mode = 'train'
    if train_mode == 'train':
        for i, (imf2, imf3) in enumerate(zip(imfs2, imfs3)):
            print('-' * 45)
            print('This is  ' + str(i) + '  time(s)')
            print('*' * 45)
            X1_train, Y1_train = data_split_test(imf2, lookback_window)
            X2_train, Y2_train = data_split_test(imf3, lookback_window)
            X_train = np.concatenate((X1_train, X2_train), axis=0)
            Y_train = np.concatenate((Y1_train, Y2_train), axis=0)
            model, history = LSTM_Model(X_train, Y_train, model_name, i)
            visualize(history, i, model_name)
    test_mode = 'test'
    if test_mode == 'test':
        for i, imf1 in enumerate(imfs1):
            print('-' * 45)
            print('This is  ' + str(i) + '  time(s)')
            print('*' * 45)
            X1_test, Y1_test = data_split_test(imf1, lookback_window)
            model = load_model(f'../checkpoints/{model_name}/{model_name}_imf' + str(i) + '_100.h5')
            prediction_Y = model.predict(X1_test)
            imfs_prediction.append(prediction_Y)
        _, Y1_test_true = data_split_test(DO1, lookback_window)
        Y_test_true = Y1_test_true.reshape(-1, 1)

    prediction = np.array(imfs_prediction).sum(axis=0)

    Y_test_true = scaler_DO1.inverse_transform(Y_test_true)
    Y_test_prediction = scaler_DO1.inverse_transform(prediction)

    rmse = format(RMSE(Y_test_true, Y_test_prediction), '.4f')
    mape = format(MAPE(Y_test_true, Y_test_prediction), '.4f')
    r2 = format(r2_score(Y_test_true, Y_test_prediction), '.4f')
    mae = format(mean_absolute_error(Y_test_true, Y_test_prediction), '.4f')

    plot_curve(Y_test_true, Y_test_prediction, rmse, mae, mape, r2, 555, f'{train_mode}', model_name)

    print('RMSE:' + str(rmse) + '\n' + 'MAE:' + str(mae) + '\n' + 'MAPE:' + str(mape) + '\n' + 'R2:' + str(r2))
    eval_indicator = pd.DataFrame({'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}, index=['values'])
    eval_indicator.to_csv(f'../pictures/{model_name}/eval_indicator.csv', index=False)
