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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from pandas import concat
from PyEMD import EEMD, EMD

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
from Code.utils import plot_curve, isolutionforest, data_split, RMSE, MAPE, imf_data, data_split_LSTM


def visualize(history):
    plt.rcParams['figure.figsize'] = (10.0, 6.0)
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    os.makedirs('../pictures/EMD_LSTM', exist_ok=True)
    plt.savefig('../pictures/EMD_LSTM/Model_loss.png')
    # plt.show()


def LSTM_Model(X_train, Y_train, i):
    os.makedirs('../checkpoints/EMD_LSTM', exist_ok=True)
    filepath = "../checkpoints/EMD_LSTM/EMD_LSTM_imf" + str(i) + "_{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='auto', period=10)
    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))  # 已经确定11步长
    '''
    如果设置return_sequences = True，该LSTM层会返回每一个time step的h，
    那么该层返回的就是1个由多个h组成的2维数组了，如果下一层不是可以接收2维数组
    的层，就会报错。所以一般LSTM层后面接LSTM层的话，设置return_sequences = True，
    如果接全连接层的话，设置return_sequences = False。
    '''
    model.add(LSTM(units=32))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=2, shuffle=True,
              callbacks=[checkpoint])
    return model


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
    scaler_DO = MinMaxScaler(feature_range=(0, 1))
    DO = scaler_DO.fit_transform(DO)

    # 孤立森林
    DO = isolutionforest(DO)

    emd = EMD()
    imfs = emd.emd(DO.reshape(-1), None, 8)
    c = int(len(DO) * .85)
    lookback_window = 11
    imfs_prediction = []

    i = 1
    for imf in imfs:
        plt.subplot(len(imfs), 1, i)
        plt.plot(imf)
        i += 1

    os.makedirs('../pictures/EMD_LSTM', exist_ok=True)
    plt.savefig('../pictures/EMD_LSTM/result_imf.png')

    Y_test_true = np.zeros([len(DO) - c - lookback_window, 1])

    train_mode = True
    if train_mode:
        i = 1
        for imf in imfs:
            print('-' * 45)
            print('This is  ' + str(i) + '  time(s)')
            print('*' * 45)
            X1_train, Y1_train, X1_test, Y1_test = data_split(imf_data(imf, 1), c, lookback_window)
            X2_train, Y2_train, X2_test, Y2_test = data_split_LSTM(X1_train, Y1_train, X1_test, Y1_test)
            Y_test_true += Y2_test
            model = LSTM_Model(X2_train, Y2_train, i)
            # model.save('EEMD_LSTM-imf' + str(i) + '.h5')
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
            model = load_model("../checkpoints/EMD_LSTM/EMD_LSTM_imf" + str(i) + "_{epoch:02d}.h5")
            prediction_Y = model.predict(X2_test)
            imfs_prediction.append(prediction_Y)
            i += 1

    imfs_prediction = np.array(imfs_prediction)
    prediction = [0.0 for i in range(len(Y_test_true))]
    prediction = np.array(prediction)
    for i in range(len(Y_test_true)):
        t = 0.0
        for imf_prediction in imfs_prediction:
            t += imf_prediction[i][0]
        prediction[i] = t

    Y_test_prediction = prediction.reshape(prediction.shape[0], 1)

    Y_test_true = scaler_DO.inverse_transform(Y_test_true)
    Y_test_prediction = scaler_DO.inverse_transform(Y_test_prediction)
    rmse = format(RMSE(Y_test_true, Y_test_prediction), '.4f')
    mape = format(MAPE(Y_test_true, Y_test_prediction), '.4f')
    r2 = format(r2_score(Y_test_true, Y_test_prediction), '.4f')
    mae = format(mean_absolute_error(Y_test_true, Y_test_prediction), '.4f')

    plot_curve(Y_test_true, Y_test_prediction, rmse, mae, mape, r2,  5, 'test', 'EMD_LSTM')
    print('RMSE:' + str(rmse) + '\n' + 'MAE:' + str(mae) + '\n' + 'MAPE:' + str(mape) + '\n' + 'R2:' + str(r2))
    eval_indicator = pd.DataFrame({'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}, index=['values'])
    eval_indicator.to_csv('../pictures/EMD_LSTM/eval_indicator.csv', index=False)
