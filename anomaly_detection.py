# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 使用第几个GPU， 0是第一个
import matplotlib.pyplot as plt
import numpy as np
import time
import keras
import pandas as pd
from sklearn.metrics import r2_score
from data_split import get_split_prep_data
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM,GRU
from keras.models import Sequential,load_model
from keras.utils.vis_utils import plot_model
from keras import regularizers

sequence_length = 99 # 制定输入的每一个序列的长度
epochs =5 # 迭代次数
batch_size = 100  # 一次样本返回数量
l2zhenze=0.00001

trst=0
tred=2000
tsst=100
tsed=2500

model_path='model/MODEL7.hdf5'#save model

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def mape_t(y_test,y_pred):
    mapeval=(np.sum(abs((y_test-y_pred)/y_test)))/len(y_test)
    return mapeval

def R2(y_test, y_pred):
    SStot = np.sum((y_test - np.mean(y_test)) ** 2)
    SSres = np.sum((y_test - y_pred) ** 2)
    r2 = 1 - SSres / SStot
    return r2

def build_model():  # LSTM模型
    model = Sequential()
    layers = {'input': 8, 'hidden1': 32, 'hidden2': 300,'hidden3': 200, 'hidden4': 50, 'output': 8}  # net参数设置 每一个值的含义 字典有label和data

    model.add(LSTM(
        input_length=sequence_length,
        input_dim=layers['input'],
        output_dim=layers['hidden1'],  # 隐藏层1
        return_sequences=True,
        ))
    model.add(Dropout(0.2))

    # model.add(LSTM(  # 隐藏层3
    #      layers['hidden2'],
    #      return_sequences=True,
    #
    #      ))
    # model.add(Dropout(0.2))

    model.add(LSTM(  # 隐藏层3
         layers['hidden3'],
         return_sequences=False,
        bias_regularizer=regularizers.l2(l2zhenze),
        kernel_regularizer=regularizers.l2(l2zhenze),
        activity_regularizer=regularizers.l2(l2zhenze),
         ))
    model.add(Dropout(0.2))

    # model.add(Dense(  # 隐藏层3
    #     layers['hidden4'],
    #
    #     ))
    # model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers['output']))  #
    model.add(Activation("linear"))  # linear

    start = time.time()
    model.compile(loss="mse", optimizer="adam")  # 损失函数mse 优化器rmsprop 损失函数xiangdui为相对误差
    print("Compilation Time : ", time.time() - start)
    model.summary()
    return model

def run_network(model=None, data=None):
    global_start_time = time.time()

    if data is None:
        print('Loading data... ')
        X_train, y_train, X_test, y_test = get_split_prep_data(trst,tred,tsst,tsed, sequence_length)
    else:
        X_train, y_train, X_test, y_test = data

    print('\nData Loaded. Compiling...\n')

    if model is None:
        model = build_model()

    plot_model(model, to_file='model1.png', show_shapes=True)
    history = LossHistory()
    print("Training...")
    model.fit(  # https://www.cnblogs.com/bymo/p/9026198.html 自动切分数据
        X_train, y_train,
        batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test)
        , callbacks=[history])  # 0.05, validation_split=0.05,validation_data=(X_test, y_test)
    # 绘制acc-loss曲线
    history.loss_plot('epoch')

    print("Predicting...")
    predicted = model.predict(X_test)
    model.save(model_path)

    print("Reshaping predicted")
    # predicted = np.reshape(predicted, (predicted.size,))
    try:
        plt.figure(1)
        # y_test=y_test.T
        # predicted=predicted.T
        error=[]
        j=421
        k=[0,1,2,3]
        for i in k:
            plt.subplot(j)
            if i==0:
                plt.title("raw data/predicted data")
            plt.plot(y_test[:, i], 'b', label='real data')
            # plt.legend()
            plt.plot(predicted[:, i], 'r--', label='predicted data')
            # plt.legend()
            j += 1
            plt.subplot(j)
            mae = abs(y_test[:, i] - predicted[:, i])
            mae=mae.T
            if i==0:
                plt.title("MAE")
            plt.plot(mae, 'r')
            j += 1
            error.append(mae)
        plt.show()

        plt.figure(2)
        j=421
        k=[4,5,6,7]
        for i in k:
            plt.subplot(j)
            if i==4:
                plt.title("raw data/predicted data")
            plt.plot(y_test[:, i], 'b', label='real data')
            # plt.legend()
            plt.plot(predicted[:, i], 'r--', label='predicted data')
            # plt.legend()
            j += 1
            plt.subplot(j)
            mae = abs(y_test[:, i] - predicted[:, i])
            mae=mae.T
            if i==4:
                plt.title("MAE")
            plt.plot(mae, 'r')
            j += 1
            error.append(mae)
        plt.show()
        error=np.array(error)

        csvFile = pd.DataFrame()
        for csvt in range(8):
            csvFile[csvt] = error[csvt]  # mse 的数据
        csvFile.to_csv("mae_data.csv")

        # #模型评估
        # r2 = R2(y_test, predicted)
        # print('Test R2: %.3f' % r2)
        # valloss=np.sum((y_test-predicted)**2)/len(y_test)
        # rmse=(valloss)**0.5
        # print('Test MSE: %.3f' % valloss)
        # print('Test RMSE: %.3f' % rmse)
        # mapee=mape_t(y_test,predicted)
        # print('Test mape: %.3f' % mapee)

        # csvFile = pd.DataFrame()
        # csvFile['pre'] = predicted
        # csvFile['test'] = y_test
        # csvFile['mse'] = mae
        # csvFile.to_csv("si_pre_data.csv")


    except Exception as e:
        print("plotting exception")
        print(str(e))
    print('Training duration (s) : ', time.time() - global_start_time)
    return model, y_test, predicted


run_network()
