# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/21 5:43
# @Function:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from math import sqrt

# 数据预处理和模型配置
n_steps = 60  # 使用更长的时间步长
n_features = 1

# 加载数据
data = pd.read_csv('../data/Bitcoin_Price/bitcoin_2018-01-01_2020-12-31.csv')
data['Date'] = pd.to_datetime(data['Start'])
data.set_index('Date', inplace=True)

# 数据标准化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# 准备输入输出对
X, y = [], []
for i in range(len(data_scaled) - n_steps):
    X.append(data_scaled[i:i+n_steps])
    y.append(data_scaled[i+n_steps])
X, y = np.array(X), np.array(y)

# 划分数据集
split = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# LSTM模型构建
model = Sequential([
    LSTM(100, activation='relu', input_shape=(n_steps, n_features)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, verbose=1)

# 预测
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# 实际值与预测值对比
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 性能评估
mse = mean_squared_error(actual, predictions)
rmse = sqrt(mse)
mae = mean_absolute_error(actual, predictions)

print("Performance Evaluation:")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)

# 可视化聚焦于最近的数据点
plt.figure(figsize=(12, 6))
plt.plot(actual[-180:], label='Actual Prices', color='blue')  # 只显示最后180天的数据
plt.plot(predictions[-180:], label='Predicted Prices', color='red')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
