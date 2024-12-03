# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/21 1:53
# @Function:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.dates import DateFormatter, AutoDateLocator

# 读取数据
data = pd.read_csv('../data/Bitcoin_Price/bitcoin_2021-02-05_2022-12-27.csv')
data['Start'] = pd.to_datetime(data['Start'])
data.set_index('Start', inplace=True)

# 计算对数收益率
data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

# 重新采样数据以获得不同周期的平均收盘价和对数收益率
resampled_data = {
    'Daily': data['Close'].resample('D').mean(),
    'Weekly': data['Close'].resample('W').mean(),
    'Monthly': data['Close'].resample('ME').mean(),
    'Quarterly': data['Close'].resample('QE').mean(),

}

log_returns_resampled = {
    'Daily': data['Log_Returns'].resample('D').mean(),
    'Weekly': data['Log_Returns'].resample('W').mean(),
    'Monthly': data['Log_Returns'].resample('ME').mean(),
    'Quarterly': data['Log_Returns'].resample('QE').mean(),

}

# 季节性分解配置
periods = {'Daily': 1, 'Weekly': 7, 'Monthly': 30, 'Quarterly': 91}

# 创建子图
fig, axes = plt.subplots(nrows=len(periods), ncols=1, figsize=(10, 20))

# 季节性分解
for i, (key, period) in enumerate(periods.items()):
    result = seasonal_decompose(data['Close'].dropna(), model='additive', period=period)
    axes[i].plot(result.trend, label='Trend')
    axes[i].plot(result.seasonal, label='Seasonal', linestyle='--')
    axes[i].plot(result.resid, label='Residual', linestyle=':')
    axes[i].set_title(f'Seasonal Decomposition - {key}')
    axes[i].legend()

fig.tight_layout()
plt.show()

# 绘制每个周期的平均收盘价和对数收益率的时间序列图与直方图
fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(18, 30))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

date_format = DateFormatter("%Y-%m")  # 设置时间格式为年-月
locator = AutoDateLocator()  # 自动找到日期位置

for i, (key, values) in enumerate(resampled_data.items()):
    # 平均收盘价时间序列图
    ax = axes[2*i, 0]
    ax.plot(values.index, values, label=f'{key} Average Close Price', color='blue')
    ax.set_title(f'{key} Average Close Price Time Series')
    ax.set_ylabel('Average Close Price')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(date_format)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    ax.legend()

    # 对数收益率时间序列图
    ax = axes[2*i+1, 0]
    ax.plot(values.index, log_returns_resampled[key], label=f'{key} Log Returns', color='green')
    ax.set_title(f'{key} Log Returns Time Series')
    ax.set_ylabel('Log Returns')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(date_format)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    ax.legend()

    # 对数收益率直方图
    ax_hist = axes[2*i, 1]
    ax_hist.hist(log_returns_resampled[key].dropna(), bins=50, alpha=0.7, color='red')
    ax_hist.set_title(f'{key} Log Returns Histogram')
    ax_hist.set_xlabel('Log Returns')
    ax_hist.set_ylabel('Frequency')
    ax_hist.grid(True)

    # 空白图用来占位保持格式整齐
    axes[2*i+1, 1].axis('off')

plt.show()
