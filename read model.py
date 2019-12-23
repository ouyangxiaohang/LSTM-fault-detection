import matplotlib.pyplot as plt
import numpy as np
import time
import keras
import pandas as pd
from keras import losses
from data_split import get_split_prep_data
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential,load_model
from keras.utils.vis_utils import plot_model
from keras import regularizers

sequence_length = 99# 制定输入的每一个序列的长度
epochs =80 # 迭代次数
batch_size = 20  # 一次样本返回数量
l2zhenze=0.00001
start_test=100
end_test=2500
model_path='model/MODEL7.hdf5'#save model

def R2(y_test, y_pred):
    SStot = np.sum((y_test - np.mean(y_test)) ** 2)
    SSres = np.sum((y_test - y_pred) ** 2)
    r2 = SSres / SStot
    return 1-r2

def R2_1_(y_test,y_pred):
    SStot=np.sum((y_test-np.mean(y_test))**2)
    SSres=np.sum((y_test-y_pred)**2)
    r2=SSres/SStot
    return 1-r2

def mape_t(y_test,y_pred):
    mapeval=(np.sum(abs((y_test-y_pred)/y_test)))/len(y_test)
    return mapeval

def run_network(model=None, data=None):
    global_start_time = time.time()

    if data is None:
        print('Loading data... ')
        X_train, y_train, X_test, y_test = get_split_prep_data(0, 1000,start_test, end_test, sequence_length)

    else:
        X_train, y_train, X_test, y_test = data

    print('\nData Loaded. Compiling...\n')

    if model is None:
        print('error model not found')


    print("Predicting...")
    predicted = model.predict(X_test)

    print("Reshaping predicted")

    try:
        plt.figure(1)
        # y_test=y_test.T
        # predicted=predicted.T
        error=[]
        j=421
        k=[0,1,2,3,4,5,6,7]
        for i in k:
            plt.subplot(j)
            # if i==0:
                # plt.title("raw data/predicted data")
            # plt.ylim(-0.5 , 1.1)
            plt.plot(y_test[:, i], 'b', label='real data')
            # plt.legend()
            plt.plot(predicted[:, i], 'r--', label='predicted data')
            # plt.legend()
            j += 1
            # plt.subplot(j)
            # mae = abs(y_test[:, i] - predicted[:, i])
            # mae=mae.T
            # if i==0:
            #     plt.title("MAE")
            # plt.plot(mae, 'r')
            # j += 1
            # error.append(mae)
        plt.show()

        # plt.figure(2)
        # j=421
        # k=[4,5,6,7]
        # for i in k:
        #     plt.subplot(j)
        #     if i == 4:
        #         plt.title("raw data/predicted data")
        #     plt.plot(y_test[:, i], 'b', label='real data')
        #     # plt.legend()
        #     plt.plot(predicted[:, i], 'r--', label='predicted data')
        #     # plt.legend()
        #     j += 1
        #     plt.subplot(j)
        #     mae = abs(y_test[:, i] - predicted[:, i])
        #     mae=mae.T
        #     if i==4:
        #         plt.title("MAE")
        #     plt.plot(mae, 'r')
        #     j += 1
        #     error.append(mae)
        # plt.show()
        # error=np.array(error)

        # #模型评估
        # r2 = R2(y_test, predicted)
        # print('Test R2: %.3f' % r2)
        # valloss=np.sum((y_test-predicted)**2)/len(y_test)
        # rmse=(valloss)**0.5
        # print('Test MSE: %.3f' % valloss)
        # print('Test RMSE: %.3f' % rmse)
        # mapee=mape_t(y_test,predicted)
        # print('Test mape: %.3f' % mapee)

        csvFile = pd.DataFrame()
        for csvt in range(8):
            csvFile[csvt] = error[csvt]  # mse 的数据
        csvFile.to_csv("mae_data.csv")

    except Exception as e:
        print("plotting exception")
        print(str(e))
    print('Training duration (s) : ', time.time() - global_start_time)
    return model, y_test, predicted

run_network(model = load_model(model_path),)
