# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/21 1:53
# @Function:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import signal

# 读取数据
data = pd.read_csv('../data/Bitcoin_Price/bitcoin_2021-02-05_2022-12-27.csv')
data['Date'] = pd.to_datetime(data['Start'])
data.set_index('Date', inplace=True)

# 准备线性回归模型
X = np.array(range(len(data))).reshape(-1, 1)  # 天数作为自变量
y = data['Close'].values  # 比特币收盘价作为因变量

# 拟合线性模型
model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)

# 去除趋势
detrended = y - trend

# 绘制原始数据和趋势线
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(data.index, y, label='Original')
plt.plot(data.index, trend, label='Trend', color='red')
plt.legend()
plt.title('Original Data and Linear Trend')

# 绘制去趋势后的数据
plt.subplot(212)
plt.plot(data.index, detrended, label='Detrended')
plt.legend()
plt.title('Detrended Data')

plt.tight_layout()
plt.show()

# 检查去趋势数据的结构
plt.figure(figsize=(12, 6))
plt.acorr(detrended, maxlags=20, usevlines=True)
plt.title('Autocorrelation of Detrended Data')
plt.show()


# Create a copy of the original data
extended_data = data.copy()

# Add the detrended data as a new column
extended_data['Detrended'] = detrended

# Save the extended data to a new CSV file
extended_data.to_csv('../data/detrended/detrended.csv')
