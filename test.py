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
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch
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

# def load_data(batch_size):
#     # 加载原始数据
#     df_testA = pd.read_csv(
#         './dataset/testA.csv')
#     # 拆解heartbeat_signals
#     test_signals = np.array(df_testA['heartbeat_signals'].apply(
#         lambda x: np.array(list(map(float, x.split(','))), dtype=np.float32)))
#     test_labels = np.empty(len(test_signals))
#     # 构建pytorch数据类
#     test_data = MyData(test_signals,test_labels)
#     # 构建pytorch数据集Dataloader
#     test_loader = Data.DataLoader(
#         dataset=test_data, batch_size=batch_size, shuffle=True)
#     return test_data, test_loader
model_path = "./HeartBeatClassification.pth"  
model = model_CNN_1()
model = torch.load(model_path, weights_only=False)
model.eval()

