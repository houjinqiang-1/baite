import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
# df = pd.DataFrame(data)

# data_sun = pd.read_csv("new2.csv",engine='python')
# print(data_sun)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0124.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
data = one_row_df

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0314.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0321.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0328.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0411.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0418.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0425.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0509.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0516.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0523.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0530.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0606.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0620.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0627.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0704.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0711.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0718.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0725.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0801.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0808.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0815.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0822.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0829.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0905.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-0912.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-1017.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-1024.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-1031.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-1107.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-1114.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-1121.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-1128.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-1205.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-1212.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-1219.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\matrix-1226.csv")
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

data = data.drop(data.columns[:0],axis = 1)
or_data = data
print(data)

data.to_csv("new1.csv")




# diff_matrix矩阵
df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0125-0124.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data = one_row_df

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0315-0314.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0322-0321.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0329-0328.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0412-0411.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0419-0418.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0426-0425.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0510-0509.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0517-0516.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0524-0523.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0531-0530.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0607-0606.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0621-0620.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0628-0627.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0705-0704.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0712-0711.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0719-0718.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0726-0725.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0802-0801.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0809-0808.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0816-0815.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0823-0822.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0830-0829.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0906-0905.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-0913-0912.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-1018-1017.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-1025-1024.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-1101-1031.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-1108-1107.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-1115-1114.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-1122-1121.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-1129-1128.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-1206-1205.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-1213-1212.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-1220-1219.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

df = pd.read_csv(r"D:\文件夹\2021data-lasso回归\diff_matrix-1227-1226.csv")
row,col = df.shape
for i in range(0,row):
    for j in range(0,col):
        if df.iloc[i,j] == -999999999:
            df.iloc[i,j] = 0
df = df.set_index('Unnamed: 0')
# 将DataFrame转置并转换为一行
one_row_df = df.T.stack().to_frame().T
# 取第一行
data1 = one_row_df.iloc[0]
data = data._append(data1)

print(data)

data.to_csv("new2.csv")




# lasso

train_Y = data.iloc[:,10000]
print(train_Y)
train_X = or_data
print(train_X)
regr = linear_model.LassoCV(alphas = [50000],max_iter=100000)
regr.fit(train_X, train_Y)

pre_Y = regr.predict(train_X)
r2 = r2_score(pre_Y , train_Y)
print(train_Y)
print(pre_Y)
print(r2)

plt.scatter(train_Y, pre_Y)
plt.show()

print('lasso系数=', regr.alpha_)
print('模型参数', regr.coef_)
coef = regr.coef_
# coef = np.insert(coef, 2, -1)
# coef_new = coef.reshape(1,-1)

arr = pd.DataFrame(coef ,index = or_data.columns)
print('相关系数数组为\n', arr)
arr.to_csv("aaa.csv")
print(arr.shape)
# 返回相关系数是否为0的布尔数组
mask = coef != 0.0
# 对特征进行选择
newdata = data.loc[:, mask]
# newdata.to_excel('lasso_text.xlsx')
newdata.to_csv("lasso10000.csv")




# print("原始DataFrame：")
# print(df)
# print("\n转换后的一行DataFrame：")
# print(one_row_df)
# print(one_row_df.shape)
# print(type(one_row_df))
# data.to_csv("new2.csv")