#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear regression
batch Gradient descent
Least square method

X: n 维特征
Y：1 维
"""

__author__ = 'Richard_l'

import numpy as np
import time

A = []  # 待估参数向量
normalization_data = None  # 归一化参数的二维数组，存储训练数据每一维的平均数(mean)和(max-min),所以应该是2*n的数组，n为数据的特征维数

# data：二维数组(float)，原始训练数据，如[[1,2,3],[2,2,1],[4,5,8]]，表示的是特征为二维，此处最后一列3,1,8为y值
# speed：float类型，learning rate，学习速度，步长,如0.001, 0.003, 0.01, 0.03, 0.1, ...设置过大会跳过最优值,按照3倍递增的方式逐步提高学习率，观察效果
# threshold：float类型，收敛阈值，越小线性拟合程度越好，但速度越慢；可设置为0，拟合程度最好
# 如果特征维度数不超过10000，speed和threshold对结果没影响，可不调节
def LR(train_data=[], speed=0.001, threshold=0.001):
    if len(train_data) == 0:
        print("train_data is null")
        return

    global A, normalization_data

    m = len(train_data)  # 训练集大小
    n = len(train_data[0]) - 1  # 维数
    X = np.zeros([m, n + 1])  # 变量的矩阵表示，维数加1，第一列用1填充

    normalization_data = np.zeros([2, n])
    # 特征归一化到区间[-1,1],加快收敛速度
    ndata = np.array(train_data)
    for i in range(n):
        trans_data = feature_scaling(i, list(ndata[:, i]))
        for j in range(m):
            train_data[j][i] = trans_data[j]

    for i in range(m):
        for j in range(n + 1):
            if j == 0:
                X[i][j] = 1
            else:
                X[i][j] = train_data[i][j-1]
    Y = np.zeros([m, 1])
    for i in range(m):
        Y[i][0] = train_data[i][n]

    A = np.zeros([n + 1, 1])  # 待估参数,初始化为参数为0向量
    if n <= 10000:  # 当维数不超过10000时，求逆矩阵较快，选择使用最小二乘法
        A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    else:  # 特征维数超过10000时，梯度下降法更快
        old_square_sum = 0
        for i in range(m):
            y_pre = 0
            for j in range(n + 1):
                y_pre += (A[j][0] * X[i][j])
            old_square_sum += (y_pre - Y[i][0]) ** 2
        old_square_sum *= 0.5  # 初始参数向量时的平方差损失度

        count = 0  # 计数，收敛次数
        while True:
            bias_y = [None for i in range(m)]  # 当参数为A时，预测值与实际值的差值列表
            for i in range(n + 1):
                bias_sum = 0
                for j in range(m):
                    if bias_y[j] is None:
                        y_pre = 0
                        for k in range(n + 1):
                            y_pre += (A[k][0] * X[j][k])
                        bias_y[j] = y_pre - Y[j][0]
                    bias_sum += bias_y[j] * X[j][i]
                A[i][0] -= speed * bias_sum
            # print(A)

            square_sum = 0
            for i in range(m):
                y_pre = 0
                for j in range(n + 1):
                    y_pre += (A[j][0] * X[i][j])
                square_sum += (y_pre - Y[i][0])**2
            square_sum *= 0.5

            count += 1
            print("收敛%d次" % count)
            print('old_square_sum:', old_square_sum)  # 参数A调整前整体的平方差
            print('square_sum:', square_sum)  # 参数A调整后整体的平方差
            print('old_square_sum-square_sum:', (old_square_sum-square_sum))  # 可观察每次收敛的平方差是否依次减小，否则异常，可调整学习速度speed
            print('================================================')
            if abs(old_square_sum - square_sum) <= threshold:
                break
            old_square_sum = square_sum

    y = ""
    for i in range(n + 1):
        if i == 0:
            y += str(A[i][0]) + "+"
        else:
            y += str(A[i][0]) + "*x" + str(i) + "+"
    model = "y=%s" % y[0:len(y)-1]
    print("model is: %s (%s)" % (model, "特征数据经过归一化((x-mean)/(max-min))后的模型)"))

# 对特征归一化
# data：list，一个维度上的特征
def feature_scaling(dim=0, data=[]):
    if len(data) == 0:
        return data
    d_max = data[0]
    d_min = data[0]
    d_sum = 0
    for one in data:
        d_sum += one
        if d_max < one:
            d_max = one
        if d_min > one:
            d_min = one
    mean = d_sum / len(data)
    normalization_data[0][dim] = mean
    normalization_data[1][dim] = d_max - d_min
    for i in range(len(data)):
        data[i] = (data[i] - mean) / (d_max - d_min)
    return data

# X list，一维特征数组
# A list,参数列表
def predict(X=[]):
    global A, normalization_data
    if len(X) != len(A)-1:
        print("data is not normal")
        return
    for i in range(len(X)):
        X[i] = (X[i] - normalization_data[0][i]) / normalization_data[1][i]
    y = 0
    X.insert(0, 1)
    for i in range(len(X)):
        y += (A[i][0] * X[i])
    print("the result of prediction is:", y)
    return y


# example
start = time.time()
data = [
    [2104, 3, 400],  # X=[2104,3], y=400
    [1600, 3, 330],
    [2400, 3, 369],
    [1416, 2, 232],
    [3000, 4, 540]
]

# 如果特征维度数不超过10000，speed和threshold对结果没影响，可不设置，直接调用LR(data)
speed = 0.1
threshold = 0.001

LR(data, speed, threshold)
predict([4000, 6])  # X=[4000, 6]
end = time.time()
print("time consumed：", (end - start), "sec")
