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
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.layers import Activation

import math
import matplotlib.pyplot as plt

from keras import Input, Model
from keras.layers import Dense
from keras.models import load_model
from Code.utils import plot_curve, data_split, isolutionforest, imf_data, data_split_LSTM, RMSE, MAPE, data_split_2, \
    data_split_test


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
    model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=2, shuffle=True,
              callbacks=[checkpoint])
    return model


if __name__ == '__main__':
    model_name = 'EEMD_LSTM_3_TP'
    data_name = 'TP_hlx'
    plt.rcParams['figure.figsize'] = (10.0, 5.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    dataset = pd.read_csv('../data/test_0826.csv', header=0, index_col=0, parse_dates=True)
    DO = dataset[data_name].values.reshape(-1, 1)
    # data = dataset.values.reshape(-1)

    # values = dataset.values
    # groups = [0, 1, 2, 3]
    # fig, axs = plt.subplots(1)

    # df = pd.DataFrame(dataset)  # 整体数据的全部字典类型
    # do = df['NH4_hlx']  # 返回溶解氧那一列，用字典的方式

    # DO = do.values.reshape(do.shape[0], 1)

    # DO = []
    # for i in range(0, len(do)):
    #     DO.append([do[i]])
    # tmp = DO

    # 孤立森林
    DO = isolutionforest(DO)

    # # 归一化
    # scaler_DO = MinMaxScaler(feature_range=(0, 1))
    # DO = scaler_DO.fit_transform(DO)

    # 标准化
    scaler_DO = StandardScaler()
    DO = scaler_DO.fit_transform(DO)

    tmp = DO
    eemd = EEMD()
    eemd.noise_seed(12345)
    imfs = eemd.eemd(DO.reshape(-1), None, 8)
    print('imfs length: ', len(imfs))

    # print(imfs.shape)
    c = int(len(DO) * 1)
    lookback_window = 11
    imfs_prediction = []

    i = 1
    for imf in imfs:
        plt.subplot(len(imfs), 1, i)
        plt.plot(imf)
        i += 1

    os.makedirs(f'../pictures/{model_name}', exist_ok=True)
    plt.savefig(f'../pictures/{model_name}/result_imf.png')

    Y_test_true = np.zeros([len(DO) - lookback_window, 1])

    train_mode = 'test'
    i = 1
    # model_name = 'EEMD_LSTM_3'
    for imf in imfs:
        print('-' * 45)
        print('This is  ' + str(i) + '  time(s)')
        print('*' * 45)
        X_test, _ = data_split_test(imf, lookback_window)
        # Y_test_true += Y2_test
        model = load_model(f'../checkpoints/{model_name}/{model_name}_imf' + str(i) + '_100.h5')
        prediction_Y = model.predict(X_test)

        imfs_prediction.append(prediction_Y)
        i += 1
    _, Y_test_true1 = data_split_test(DO, lookback_window)
    Y_test_true = Y_test_true1.reshape(Y_test_true1.shape[0], 1)
    # model_name = 'EEMD_LSTM_test'

    imfs_prediction = np.array(imfs_prediction)
    prediction = imfs_prediction.sum(axis=0)
    # prediction = [0.0 for i in range(len(Y_test_true))]
    # prediction = np.array(prediction)
    # for i in range(len(Y_test_true)):
    #     t = 0.0
    #     for imf_prediction in imfs_prediction:
    #         t += imf_prediction[i][0]
    #     prediction[i] = t
    #
    # prediction = prediction.reshape(prediction.shape[0], 1)

    Y_test_true = scaler_DO.inverse_transform(Y_test_true)
    # Y_test_true1 = scaler_DO.inverse_transform(Y_test_true1)
    Y_test_prediction = scaler_DO.inverse_transform(prediction)

    rmse = format(RMSE(Y_test_true, Y_test_prediction), '.4f')
    mape = format(MAPE(Y_test_true, Y_test_prediction), '.4f')
    r2 = format(r2_score(Y_test_true, Y_test_prediction), '.4f')
    mae = format(mean_absolute_error(Y_test_true, Y_test_prediction), '.4f')

    plot_curve(Y_test_true, Y_test_prediction, rmse, mae, mape, r2, 500, data_name, model_name)

    # for i in range(0, len(Y_test_prediction), 50):
    #     plot_curve(Y_test_true[i:i + 50], Y_test_prediction[i:i + 50], rmse, mae, mape, r2, i, f'{train_mode}_{i}', model_name)

    print('RMSE:' + str(rmse) + '\n' + 'MAE:' + str(mae) + '\n' + 'MAPE:' + str(mape) + '\n' + 'R2:' + str(r2))
    eval_indicator = pd.DataFrame({'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}, index=['values'])
    eval_indicator.to_csv(f'../pictures/{model_name}/eval_indicator.csv', index=False)
