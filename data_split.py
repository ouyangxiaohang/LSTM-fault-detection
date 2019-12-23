import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyHampel
from sklearn.preprocessing import MinMaxScaler
random_data_dup =5 # each sample randomly duplicated between 0 and 9 times, see dropin function每个样本随机重复0 - 9次，见dropin函数

def dropin(X, y):
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    X_hat = []  # x的预测
    y_hat = []  # y的预测
    for i in range(0, len(X)):
        for j in range(0, np.random.random_integers(0, random_data_dup)):
            X_hat.append(X[i, :])
            y_hat.append(y[i])
    return np.asarray(X_hat), np.asarray(y_hat)

#归一化
def Normalize(list):
    list = np.array(list)
    low, high = np.percentile(list, [0, 100])
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = (list[i]-low)/delta
    return  list

#去除极端值
def qujiduan(data):
    for i in range(len(data)):
        if data[i]>=0.9:
            data[i]=data[i-1]
    return data

def z_norm(result):         #Z-score标准化方法
    result_mean = result.mean() #平均值
    result_std = result.std() #标准差非样本差
    result -= result_mean
    result /= result_std
    return result

def norm_T(result,result_std,result_mean):
    result*=result_std
    result+=result_mean
    return result

#将所有输入量按照列进行归一化
def MinMax_data(data):
    dataT = data.T  # 转置（8*）
    result = []
    for i in range(dataT.shape[0]):
        ans_data = Normalize(dataT[i])
        # ans_data = z_norm(ans_data)
        result.append(ans_data)
    result = np.array(result)
    result = result.T
    return  result

def quyichang(y):
    plt.figure(1)
    plt.plot(y,'r',label = 'raw data')
    plt.legend()
    y1,waveIdx = pyHampel.hampel(y,25 , method="center") #只对前2000需要训练的数据进行去砸去噪
    plt.plot(y1,'b',label = 'Hampel data')
    plt.legend()
    plt.show()
    return y1

# 数据分割为train与test
def get_split_prep_data(train_start, train_end,
                        test_start, test_end,sequence_length):
    datalist = pd.read_csv('raw_data.csv')#si-data 0-1范围 si-data0 原始数据 si-big2 2000的数据0-1范围
    list = datalist.values.tolist()
    rawdata = np.array(list)
    data =MinMax_data(rawdata)
    print("Length of Data", len(data))
    # train data
    print("Creating train data...")
    y_train = data[train_start + sequence_length+1:train_end+1]
    y_test = data[test_start+1:test_end+1]  # 加1是用前50时间步的8个变量状态预测下一个时刻si的值

    # y_train1 = data[train_start + sequence_length:train_end, -1]
    # y_test1 = data[test_start:test_end, -1]  # 加1是用前50时间步的8个变量状态预测下一个时刻si的值

    result = []
    data1=data
    # np.random.shuffle(data1)  # shuffles in-place 生成随机列表
    data2=data1[:]
    for index in range(train_start, train_end - sequence_length):  # 弄清楚为什么有sequence_length
        result.append(data2[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    print("Train data shape  : ", result.shape)
    X_train = result[:]
    print("Shape X_train", np.shape(X_train))

    result = []
    data3=data[:]
    for index in range(test_start-sequence_length, test_end - sequence_length):
        result.append(data3[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    print("Train data shape  : ", result.shape)
    X_test = result[:]
    print("Shape X_test", np.shape(X_test))

    print("MinMaxScaler")
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 8))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],8))

    return X_train, y_train, X_test, y_test
