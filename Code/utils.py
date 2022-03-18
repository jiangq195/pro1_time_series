from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os
import math
import numpy as np
import pandas as pd


def visualize(history, i, model_name):
    plt.figure(i)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    os.makedirs(f'../pictures/{model_name}', exist_ok=True)
    plt.savefig(f'../pictures/{model_name}/Loss_imf{i}.png')


# 画出曲线变化图
def plot_curve(true_data, predicted, rmse, mae, mape, r2, figure, type, model_name):
    plt.figure(figure)
    plt.plot(true_data, label='True data')
    plt.plot(predicted, label='Predicted data')
    plt.legend()
    plt.title(f'{type}_data_predict')
    plt.text(1, 1, 'RMSE:' + str(rmse) + '\n' + 'MAE:' + str(mae) + '\n' + 'MAPE:' + str(mape) + '\n' + 'R2:' + str(r2))
    os.makedirs(f'../pictures/{model_name}', exist_ok=True)
    plt.savefig(f'../pictures/{model_name}/{type}_predict.png')


# 画出曲线变化图
def plot_curve_1(true_data, predicted, figure, type, model_name, day):
    plt.figure(figure)
    plt.plot(true_data, label='True data')
    plt.plot(predicted, label='Predicted data')
    plt.legend()
    plt.title(f'{model_name}_{type}_data_{day}')
    os.makedirs(f'../pictures/{model_name}', exist_ok=True)
    plt.savefig(f'../pictures/{model_name}/{type}_day_{day}_result_final.png')


def plot_curve_2(true_data, figure, type, model_name):
    plt.figure(figure)
    plt.plot(true_data, label='True data')
    plt.legend()
    plt.title(f'{type}')
    os.makedirs(f'../pictures/{model_name}', exist_ok=True)
    plt.savefig(f'../pictures/{model_name}/{type}_data.png')


dataset = pd.read_csv('../data/train_0826.csv', header=0, index_col=0, parse_dates=True)
plot_curve_2(dataset['pH_hlx'], 1, 'pH_hlx', 'EEMD_LSTM_3')
plot_curve_2(dataset['DO_hlx'], 2, 'DO_hlx', 'EEMD_LSTM_3')
plot_curve_2(dataset['CODMn_hlx'], 3, 'CODMn_hlx', 'EEMD_LSTM_3')
plot_curve_2(dataset['NH4_hlx'], 4, 'NH4_hlx', 'EEMD_LSTM_3')
plot_curve_2(dataset['TP_hlx'], 5, 'TP_hlx', 'EEMD_LSTM_3')
plot_curve_2(dataset['TN_hlx'], 6, 'TN_hlx', 'EEMD_LSTM_3')


def isolutionforest(DO):
    rng = np.random.RandomState(42)
    clf = IsolationForest(random_state=rng, contamination=0.025)  # contamination为异常样本比例
    clf.fit(DO)

    DO_copy = DO
    m = 0
    pre = clf.predict(DO)
    for i in range(len(pre)):
        if pre[i] == -1:
            if i == 0 or i == len(pre) - 1:
                continue
                # DO_copy = np.delete(DO_copy, i - n, 0)
                # n += 1
            print(i + 2, DO[i])
            DO_copy[i] = (DO_copy[i - 1] + DO_copy[i + 1]) / 2
            m += 1
    print(m)
    return DO_copy


def RMSE(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    return rmse


def MAPE(Y_true, Y_pred):
    Y_true, Y_pred = np.array(Y_true), np.array(Y_pred)
    return np.mean(np.fabs((Y_true - Y_pred) / Y_true)) * 100


def data_split_1(data, train_len, lookback_window, n_out):
    train = data[:train_len]  # 标志训练集
    test = data[train_len:]  # 标志测试集

    # X1[]代表移动窗口中的10个数
    # Y1[]代表相应的移动窗口需要预测的数
    # X2, Y2 同理

    X1, Y1 = [], []
    for i in range(lookback_window + n_out - 1, len(train)):
        X1.append(train[i - lookback_window - n_out + 1:i - n_out + 1])
        Y1.append(train[i - n_out + 1:i + 1])
    Y_train = np.array(Y1)
    X_train = np.array(X1)

    X2, Y2 = [], []
    for i in range(lookback_window + n_out - 1, len(test)):
        X2.append(test[i - lookback_window - n_out + 1:i - n_out + 1])
        Y2.append(test[i - n_out + 1:i + 1])
    y_test = np.array(Y2)
    X_test = np.array(X2)

    return X_train, Y_train, X_test, y_test


def data_split_6(data, train_len, lookback_window, n_out):
    train = data[:train_len]  # 标志训练集
    test = data[train_len:]  # 标志测试集

    # X1[]代表移动窗口中的10个数
    # Y1[]代表相应的移动窗口需要预测的数
    # X2, Y2 同理

    X1, Y1 = [], []
    for i in range(lookback_window + n_out - 1, len(train)):
        X1.append(train[i - lookback_window - n_out + 1:i - n_out + 1])
        Y1.append(train[i])
    Y_train = np.array(Y1)
    X_train = np.array(X1)

    X2, Y2 = [], []
    for i in range(lookback_window + n_out - 1, len(test)):
        X2.append(test[i - lookback_window - n_out + 1:i - n_out + 1])
        Y2.append(test[i])
    y_test = np.array(Y2)
    X_test = np.array(X2)

    return X_train, Y_train, X_test, y_test


def data_split_2(data, train_len, lookback_window):
    # train = data[:train_len]  # 标志训练集
    test = data[train_len:]  # 标志测试集

    # X1[]代表移动窗口中的10个数
    # Y1[]代表相应的移动窗口需要预测的数
    # X2, Y2 同理

    # X1, Y1 = [], []
    # for i in range(lookback_window, len(train)):
    #     X1.append(train[i - lookback_window:i])
    #     Y1.append(train[i])
    # Y_train = np.array(Y1)
    # X_train = np.array(X1)

    X2, Y2 = [], []
    for i in range(lookback_window, len(test)):
        # X2.append(test[i - lookback_window:i])
        Y2.append(test[i])
    y_test = np.array(Y2)
    # X_test = np.array(X2)

    return y_test


def data_split_test(test, lookback_window):
    X2, Y2 = [], []
    for i in range(lookback_window, len(test)):
        X2.append(test[i - lookback_window:i])
        Y2.append(test[i])
    X_test = np.array(X2).reshape(len(X2), -1, 1)
    y_test = np.array(Y2).reshape(-1, 1)

    return X_test, y_test


def data_split(data, train_len, lookback_window):
    train = data[:train_len]  # 标志训练集
    test = data[train_len:]  # 标志测试集

    # X1[]代表移动窗口中的10个数
    # Y1[]代表相应的移动窗口需要预测的数
    # X2, Y2 同理

    X1, Y1 = [], []
    for i in range(lookback_window, len(train)):
        X1.append(train[i - lookback_window:i])
        Y1.append(train[i])
    Y_train = np.array(Y1)
    X_train = np.array(X1)

    X2, Y2 = [], []
    for i in range(lookback_window, len(test)):
        X2.append(test[i - lookback_window:i])
        Y2.append(test[i])
    y_test = np.array(Y2)
    X_test = np.array(X2)

    return X_train, Y_train, X_test, y_test


def data_split_LSTM_1(X_train, Y_train, X_test, Y_test):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
    Y_train = Y_train.reshape(Y_train.shape[0], -1)
    Y_test = Y_test.reshape(Y_test.shape[0], -1)
    return X_train, Y_train, X_test, Y_test


def data_split_LSTM(X_train, Y_train, X_test, Y_test):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    Y_test = Y_test.reshape(Y_test.shape[0], 1)
    return X_train, Y_train, X_test, Y_test


def imf_data(data, lookback_window):
    X1 = []
    for i in range(lookback_window, len(data)):
        X1.append(data[i - lookback_window:i])
    X1.append(data[len(data) - 1:len(data)])
    X_train = np.array(X1)
    return X_train
