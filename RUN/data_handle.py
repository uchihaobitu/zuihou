#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/4/11 15:32
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/4/11 15:32
# @File         : data_handle.py
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

# 假设你的 DataFrame 是这样的：
# df = pd.DataFrame({
#     'name': ['A', 'B', 'A', 'B', 'C', 'C'],
#     'timestamp': [1, 1, 2, 2, 1, 2],
#     'value': [10, 20, 30, 40, 50, 60]
# })
# df = pd.read_csv("metrics.norm.csv")
df = pd.read_csv("metrics_A2.csv")


# 如果 timestamp 列不是 datetime 类型，需要先转换一下
if not isinstance(df['timestamp'].dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# 使用 pivot 函数进行转换
df_pivot = df.pivot(index='timestamp', columns='name', values='value')
df_pivot.fillna(0, inplace=True)

# 由于 pivot 后 timestamp 变成了索引，如果你想将其作为普通列，可以使用 reset_index
df_pivot = df_pivot.reset_index()
df_pivot = df_pivot.drop(columns=['timestamp'])

for col in df_pivot.columns:
    df_pivot[col] = (df_pivot[col] - df_pivot[col].min()) / (df_pivot[col].max() - df_pivot[col].min())

df_pivot.fillna(0, inplace=True)
print(df_pivot.head())
df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]
df_pivot.to_csv("metrics_compare_A2.csv", index=False)

# df_faults = pd.read_csv("faults.csv")
# print(df_faults.head())

# print(df_pivot.head())
np_data = df_pivot.to_numpy()
print("np_data",np_data)
print("np_data",np_data.shape)
np.save('train_use_A2.npy', np_data)

# np.save('metrics.npy', np_data)