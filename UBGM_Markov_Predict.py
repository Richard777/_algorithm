#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediction of new dimension's unbiased grey model and markov

新维无偏灰色GM(1,1)马尔科夫预测

无偏灰色：主要适用于预测时间短，数据资料少，原始数据序列（或累加数据序列）按指数规律变化且变化速度不是很快，
        波动不大的系统对象，只需很少的几个数据即可建立模型进行预测,其预测的几何图形是一条较为平滑的曲线，
        对随机波动性大的数据序进行预测时，预测值起伏不定，影响预测准确度

马尔科夫：马尔柯夫链理论适用于预测随机波动大的动态过程，可以弥补灰色模型的缺点，但仍存在越往后预测准确度越低的问题

新维：利用灰色马尔柯夫预测的最新预测结果不断更新建模用的原始数据，不但可以保留短期预测准确度高的优点，
    而且对于中、长期预测准确度也有提高.
"""

__author__ = 'Richard_l'

import numpy as np
import math

# history_data：存储历史数据的list，如[2.2,3.5,4.3]
# m：整数类型，存储往后预测的步数
# ret：list类型，预测的结果
def predict(history_data=[], m=0, ret=[]):
    if m <= 0:
        print('--end')
        return
    n = len(history_data)
    if n <= 0:
        print('原始数据序列为空')
        return
    X0 = np.array(history_data)

    # 累加生成
    history_data_agg = [sum(history_data[0:i+1]) for i in range(n)]
    X1 = np.array(history_data_agg)

    # 计算数据矩阵B和数据向量Y
    B = np.zeros([n-1, 2])
    Y = np.zeros([n-1, 1])
    for i in range(0, n-1):
        B[i][0] = -0.5*(X1[i] + X1[i+1])
        B[i][1] = 1
        Y[i][0] = X0[i+1]

    # 计算GM(1,1)微分方程的参数a和u
    # A = np.zeros([2,1])
    A = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
    a = A[0][0]
    u = A[1][0]

    # print('a=', a)
    # print('u=', u)

    b = math.log((2-a)/(2+a))   # 自然对数
    A = (2*u)/(2+a)

    # 建立无偏灰色预测模型
    XX0 = np.zeros(n)
    XX0[0] = X0[0]
    for i in range(1, n):
        XX0[i] = A*math.exp(b*i)

    # print(X0)
    # print(XX0)

    # 求相对误差平均值
    p = 0
    for i in range(n):
        p += math.fabs((X0[i] - XX0[i])/X0[i])
    p /= n

    # print('相对误差平均值：',p)
    # print('相对精度平均值：',1-p)

    # 求绝对误差平均值
    g = 0
    for i in range(n):
        g += math.fabs((X0[i] - XX0[i]))
    g /= n

    # print('绝对误差平均值：',g)

    # 求最大相对误差
    mp = 0
    for i in range(n):
        tmp = math.fabs((X0[i] - XX0[i])/X0[i])
        if tmp > mp:
            mp = tmp

    print('最大相对误差：',mp)

    # 求最小相对误差
    lp = 1
    for i in range(1, n):
        tmp = math.fabs((X0[i] - XX0[i])/X0[i])
        if tmp < lp:
            lp = tmp

    print('最小相对误差(除去首项,因为首项一定相等)：',lp)

    # 模型精度的后验差检验
    e = 0      # 求残差平均值
    for i in range(0, n):
        e += (X0[i] - XX0[i])
    e /= n

    # print('残差平均值--', e)

    # 求历史数据平均值
    aver = 0
    for i in range(0, n):
        aver += X0[i]
    aver /= n

    # 求历史数据方差
    s12 = 0
    for i in range(0, n):
        s12 += (X0[i]-aver)**2
    s12 /= n

    # 求残差方差
    s22 = 0
    for i in range(0, n):
        s22 += ((X0[i] - XX0[i]) - e)**2
    s22 /= n

    # 求后验差比值
    C = math.sqrt(s22 / s12)

    # 求小误差概率
    cout = 0
    for i in range(0, n):
        if abs((X0[i] - XX0[i]) - e) < 0.6745*math.sqrt(s12):
            cout = cout+1
        else:
            cout = cout
    P = cout / n

    # print('C--', C)
    # print('P--', P)

    # pre_m = {}  #无偏灰色预测值
    if (C < 0.35 and P > 0.95):
        print('无偏灰色模型适用,预测精度为一级')
        # 预测精度为一级
        # pre_m = np.zeros(m)
        # for i in range(0,m):
        #     pre_m[i] = A*math.exp(b*(i+n))
        # print('经无偏灰色模型预测结果(往后m项)：')
        # print(pre_m)
    else:
        print('无偏灰色预测法不适用')
        return

    # 马尔科夫预测
    # 构建三种隐性状态 ：
    # 1.高估 2.正常 3.低估 (此处范围可根据实际情况调整)
    min = - mp - lp  # 低估状态误差下界
    normal_min = -(lp + (1 / 4) * (mp - lp))  # 低估状态误差上界or正常状态误差下界
    normal_max = lp + (1 / 4) * (mp - lp)  # 高估状态误差下界or正常状态误差上界
    max = mp + lp  # 高估状态误差上界

    # print('min--',min)
    # print('normal_min--',normal_min)
    # print('normal_max--',normal_max)
    # print('max--',max)

    # 根据已有数据统计处于各状态的数目(前n-1项)
    # 去掉最后一项，因为最后一项状态转换不可知
    status = {}
    status[0] = 0  # 高估个数
    status[1] = 0  # 正常个数
    status[2] = 0  # 低估个数

    # 存储每个item的状态信息
    dict = {}
    for i in range(n):
        if ((XX0[i]-X0[i])/X0[i] >= normal_max) and ((XX0[i]-X0[i])/X0[i] < max):
            if i != n-1:
                status[0] += 1
            dict[i] = 0 # 0代表高估状态
        elif ((XX0[i]-X0[i])/X0[i] >= normal_min) and ((XX0[i]-X0[i])/X0[i] < normal_max):
            if i != n - 1:
                status[1] += 1
            dict[i] = 1  # 1代表正常状态
        elif ((XX0[i]-X0[i])/X0[i] >= min) and ((XX0[i]-X0[i])/X0[i] < normal_min):
            if i != n - 1:
                status[2] += 1
            dict[i] = 2  # 2代表低估状态

    # 构建一步计数状态矩阵
    n_matrix = np.zeros([3,3])
    for i in range(n-1):
        if dict[i] == 0 and dict[i + 1] == 0:
            n_matrix[0][0] += 1
        elif dict[i] == 0 and dict[i + 1] == 1:
            n_matrix[0][1] += 1
        elif dict[i] == 0 and dict[i + 1] == 2:
            n_matrix[0][2] += 1
        elif dict[i] == 1 and dict[i + 1] == 0:
            n_matrix[1][0] += 1
        elif dict[i] == 1 and dict[i + 1] == 1:
            n_matrix[1][1] += 1
        elif dict[i] == 1 and dict[i + 1] == 2:
            n_matrix[1][2] += 1
        elif dict[i] == 2 and dict[i + 1] == 0:
            n_matrix[2][0] += 1
        elif dict[i] == 2 and dict[i + 1] == 1:
            n_matrix[2][1] += 1
        elif dict[i] == 2 and dict[i + 1] == 2:
            n_matrix[2][2] += 1

    # print('一步计数状态矩阵：')
    # print(n_matrix)

    # 根据计数状态矩阵构建一步状态转移矩阵
    m_matrix = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            m_matrix[i][j] = n_matrix[i][j]/status[i]

    print('一步状态转移矩阵：')
    print(m_matrix)

    # # 往后预测的item数m
    # pre_m = {}  #无偏灰色预测值
    # for i in range(m):
    #     pre_m[i] = A*math.exp(b*(i+n))

    # 经无偏灰色预测的值，第n+1项
    grey_val = A*math.exp(b*n)
    # 经马尔科夫调整过的预测值,第n+1项
    corr_val = m_matrix[dict[n - 1]][0] * 0.5 * (grey_val / (1 + normal_max) + grey_val / (1 + max)) \
               + m_matrix[dict[n - 1]][1] * 0.5 * (grey_val / (1 + normal_min) + grey_val / (1 + normal_max)) \
               + m_matrix[dict[n - 1]][2] * 0.5 * (grey_val / (1 + normal_min) + grey_val / (1 + min))
    ret.append(corr_val)
    print('grey predict value:', grey_val)
    print('predict value:', corr_val)
    history_data.pop(0)
    history_data.append(corr_val)
    predict(history_data, m-1, ret)


# example：
history_data = [11.331,11.849,13.215,15.599,16.862,19.441,20.427,19.096,21.856,21.818,22.499,22.164,23.944,25.251,40.906,50.063,53.439,
                54.814,50.441,49.271,53.292,58.729,63.508]
m = 7  # 往后预测的item数m
ret = []

# 经无偏灰色马尔科夫模型预测结果：
predict(history_data, m, ret)
print('经无偏灰色马尔科夫模型预测结果(往后', m, '项)：')
print(ret)


















