# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:49:27 2024

@author: lwang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_definition import model_CNN_1
from model_definition import HeartBeatsDataSet
import torch
from torch.utils.data import DataLoader
import time
# from train import model_CNN_1
# 检查cuda是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")
# 加载测试数据
df_testA = pd.read_csv(
    './dataset/testA.csv')
# 查看测试数据的前五条
print(df_testA.head())
# 检查数据是否有NAN数据
print(df_testA.isna().sum())
# 查看测试数据集信息
print(df_testA.info())


# ids = []
# for id, row in df_train.groupby('label').apply(lambda x: x.iloc[2]).iterrows():
#     ids.append(int(id))
#     signals = list(map(float, row['heartbeat_signals'].split(',')))
#     sns.lineplot(data=signals)

# plt.legend(ids)
# plt.show()

# 导入数据并构建数据集
test_signals = np.array(df_testA['heartbeat_signals'].apply(
    lambda x: np.array(list(map(float, x.split(','))), dtype=np.float32)))
test_labels = np.zeros(len(test_signals))
test_data = HeartBeatsDataSet(test_signals, test_labels)
test_dataloader = DataLoader(test_data, batch_size=8)

# 绘制测试数据集的前五条的折线图
for i in range(5):
    plt.plot(test_signals[i])
plt.show()
# 加载已训练好的模型
model_path = "./HeartBeatClassification.pth"
model = model_CNN_1().to(device)
model = torch.load(model_path, weights_only=False)
model.eval()
tstart = time.time()
for index, data in enumerate(test_dataloader):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    _,predictions = torch.max(model(inputs),1)
tend = time.time()
print('evaluate duration:{:.2f}s'.format(tend - tstart))
