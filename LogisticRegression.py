#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logistic regression

Newton’s method

X: n 维特征
Y：1 维
"""

__author__ = 'Richard_l'

import numpy as np
import math
import time

A = []  # 待估参数向量
normalization_data = None  # 归一化参数的二维数组，存储训练数据每一维的平均数(mean)和(max-min),所以应该是2*n的数组，n为数据的特征维数

# data：二维数组(float)，原始训练数据，如[[1,2,1],[2,2,0],[4,5,1]]，表示的是特征为二维，此处最后一列1,0,1为y值
# tol: 误差容忍度
def LgR(train_data=[], tol = 1e-4):
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
    H = np.zeros([n + 1, n + 1])  # Hessian矩阵
    Y_Pre = np.zeros([m, 1])  # 存储当A为某一估计向量时sigmoid函数的预测值
    delta = np.zeros([n + 1, 1])  # 似然函数(目标优化函数)对A的偏导

    # A为初始值的时候的预测值
    for i in range(m):
        Y_Pre[i][0] = 1 / (1 + math.exp(0 - A.T.dot(X[i].T)))

    # A为初始值的时候，似然函数(目标优化函数)对A的偏导
    for i in range(m):
        delta = (delta.T + ((Y[i][0] - Y_Pre[i][0]) * X[i])).T
    delta *= (1 / m)
    count = 0  # 计数，收敛次数
    while True:
        for i in range(n + 1):
            for j in range(i, n + 1):
                for k in range(m):
                    if Y_Pre[k][0] is not None:
                        H[i][j] += ((X[k][i] * X[k][j] * Y_Pre[k][0] * (Y_Pre[k][0] - 1)) * (1 / m))
                H[j][i] = H[i][j]
        try:
            A = A - np.linalg.inv(H).dot(delta)
        except:
            print("hessian矩阵是奇异矩阵，无逆矩阵")
            exit()

        for i in range(m):
            try:
                Y_Pre[i][0] = 1 / (1 + math.exp(0 - A.T.dot(X[i].T)))
                delta = (delta.T + ((Y[i][0] - Y_Pre[i][0]) * X[i])).T
            except:
                Y_Pre[i][0] = None
                continue
        delta *= (1 / m)

        print(delta)
        # 判断是否收敛
        flag = True  # 初始化为收敛
        for i in range(n + 1):
            if math.fabs(delta[i][0]) > tol:
                flag = False
                break
        if flag:
            break
        count += 1
        print("收敛%d次" % count)
        print('================================================')

    y = ""
    for i in range(n + 1):
        if i == 0:
            y += str(A[i][0]) + "+"
        else:
            y += str(A[i][0]) + "*x" + str(i) + "+"
    y = y.replace('+', '$')
    y = y.replace('-', '+')
    y = y.replace('$', '-')
    if y.startswith('+'):
        y = y[1:]
    model = "y=1/(1+exp(%s))" % y[0:len(y)-1]
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
    X.insert(0, 1)
    y = 1 / (1 + math.exp(0 - A.T.dot(X)))
    if y >= 0.5:
        y = 1
    else:
        y = 0
    print("the result of prediction is:", y)
    return y

def loadDataSet(dataFileName):
    dataMat = []
    fr = open(dataFileName)
    for line in fr:
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
    return dataMat

# example
start = time.time()

data = loadDataSet("data/LogisticRegression.txt")
LgR(data, 1e-4)
predict([-0.752157, 6.538620])  # X=[-0.752157, 6.538620]
end = time.time()
print("time consumed：", (end - start), "sec")


