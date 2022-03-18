import os

import numpy as np
import pandas as pd

from pandas import read_csv
from pandas import DataFrame
from datetime import datetime
from matplotlib import pyplot
from pylab import mpl

from sklearn.preprocessing import MinMaxScaler
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
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.layers import Activation

import math
import matplotlib.pyplot as plt

from keras import Input, Model
from keras.layers import Dense
from keras.models import load_model
from Code.utils import plot_curve, data_split, isolutionforest, imf_data, data_split_LSTM, RMSE, MAPE, data_split_1, \
    data_split_LSTM_1, plot_curve_1


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


def LSTM_Model(X_train, Y_train, i):
    os.makedirs('../checkpoints/EEMD_LSTM_1', exist_ok=True)
    filepath = "../checkpoints/EEMD_LSTM_1/EEMD_LSTM_imf" + str(i) + "_{epoch:02d}.h5"
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
    model.add(Dense(6))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=2, shuffle=True,
              callbacks=[checkpoint])
    return model


def main(lookback_window, DO, imfs, scaler_DO):
    # print(imfs.shape)
    c = int(len(DO) * .85)
    # lookback_window = 50
    n_out = 6
    imfs_prediction = []

    i = 1
    for imf in imfs:
        plt.subplot(len(imfs), 1, i)
        plt.plot(imf)
        i += 1

    os.makedirs('../pictures/EEMD_LSTM_1', exist_ok=True)
    plt.savefig('../pictures/EEMD_LSTM_1/result_imf.png')

    Y_test_true = np.zeros([len(DO) - c - lookback_window - n_out + 1, n_out])

    train_mode = True
    if train_mode:
        i = 1
        for imf in imfs:
            print('-' * 45)
            print('This is  ' + str(i) + '  time(s)')
            print('*' * 45)
            X1_train, Y1_train, X1_test, Y1_test = data_split_1(imf_data(imf, 1), c, lookback_window, n_out)
            X2_train, Y2_train, X2_test, Y2_test = data_split_LSTM_1(X1_train, Y1_train, X1_test, Y1_test)
            Y_test_true += Y2_test
            model = LSTM_Model(X2_train, Y2_train, i)
            # model.save('../lbw6/EEMD_LSTM-imf' + str(i) + '-100.h5')
            prediction_Y = model.predict(X2_test)
            imfs_prediction.append(prediction_Y)
            i += 1
    else:
        i = 1
        for imf in imfs:
            print('-' * 45)
            print('This is  ' + str(i) + '  time(s)')
            print('*' * 45)
            X1_train, Y1_train, X1_test, Y1_test = data_split(imf_data(imf, 1), c, lookback_window)
            X2_train, Y2_train, X2_test, Y2_test = data_split_LSTM(X1_train, Y1_train, X1_test, Y1_test)
            Y_test_true += Y2_test
            model = load_model('../checkpoints/EEMD_LSTM_1/EEMD_LSTM_imf' + str(i) + '_100.h5')
            prediction_Y = model.predict(X2_test)

            imfs_prediction.append(prediction_Y)
            i += 1

    imfs_prediction = np.array(imfs_prediction)
    prediction = imfs_prediction.sum(axis=0)

    Y_test_true = scaler_DO.inverse_transform(Y_test_true)
    Y_test_prediction = scaler_DO.inverse_transform(prediction)

    rmse = format(RMSE(Y_test_true, Y_test_prediction), '.4f')
    mape = format(MAPE(Y_test_true, Y_test_prediction), '.4f')
    r2 = format(r2_score(Y_test_true, Y_test_prediction), '.4f')
    mae = format(mean_absolute_error(Y_test_true, Y_test_prediction), '.4f')

    plot_curve_1(Y_test_true[:, 0], Y_test_prediction[:, 0], 5, 'test', 'EEMD_LSTM_1', 1)
    plot_curve_1(Y_test_true[:, 1], Y_test_prediction[:, 1], 6, 'test', 'EEMD_LSTM_1', 2)
    plot_curve_1(Y_test_true[:, 2], Y_test_prediction[:, 2], 7, 'test', 'EEMD_LSTM_1', 3)
    plot_curve_1(Y_test_true[:, 3], Y_test_prediction[:, 3], 8, 'test', 'EEMD_LSTM_1', 4)
    plot_curve_1(Y_test_true[:, 4], Y_test_prediction[:, 4], 9, 'test', 'EEMD_LSTM_1', 5)
    plot_curve_1(Y_test_true[:, 5], Y_test_prediction[:, 5], 10, 'test', 'EEMD_LSTM_1', 6)

    print('RMSE:' + str(rmse) + '\n' + 'MAE:' + str(mae) + '\n' + 'MAPE:' + str(mape) + '\n' + 'R2:' + str(r2))
    with open('../pictures/EEMD_LSTM_1/results.txt', 'a') as f:
        f.write(f'{lookback_window}' + '\t' + 'RMSE:' + str(rmse) + '\t' + 'MAE:' + str(mae) + '\t' + 'MAPE:' + str(
            mape) + '\t' + 'R2:' + str(r2) + '\n')
    eval_indicator = pd.DataFrame({'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}, index=['values'])
    eval_indicator.to_csv('../pictures/EEMD_LSTM_1/eval_indicator.csv', index=False)


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (10.0, 5.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    dataset = pd.read_csv('../data/initial_0826.csv', header=0, index_col=0, parse_dates=True)
    data = dataset.values.reshape(-1)

    values = dataset.values
    groups = [0, 1, 2, 3]
    # fig, axs = plt.subplots(1)

    df = pd.DataFrame(dataset)  # 整体数据的全部字典类型
    do = df['NH4_hlx']  # 返回溶解氧那一列，用字典的方式

    DO = []
    for i in range(0, len(do)):
        DO.append([do[i]])
    tmp = DO
    scaler_DO = MinMaxScaler(feature_range=(0, 1))
    DO = scaler_DO.fit_transform(DO)

    # 孤立森林
    DO = isolutionforest(DO)

    eemd = EEMD()
    eemd.noise_seed(12345)
    imfs = eemd.eemd(DO.reshape(-1), None, 8)
    for i in range(11, 100, 2):
        main(i, DO, imfs, scaler_DO)
