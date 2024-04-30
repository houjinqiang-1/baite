# encoding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
def data_processing(data):
    print("We will analyze and test",len(data),"personnel,including",len(data.columns),"attribute columns")
    data = data.replace(" ", pd.NA)#因表中的空值为空格，将表中的空格转化为NA
    #针对人员，去除缺失率高的数据
    row,col = data.shape
    miss_row = 0
    # print(row,col)
    for i in range(row):
        count = data.loc[i,:].isnull().sum()#第i行缺失总数
        rate = count/(col - 1)#-1表示去掉人员编号列
        if rate > 0.1:#对缺失率过高的人员数据进行删除
            data = data.drop(index = i)
            miss_row += 1
    # print(data)
    # print('Because the missing rate is too high, delete the ',miss_row,'-person information')

    #删除缺失率较高的属性列
    delCol = []
    miss_col = 0
    for col in data.columns:
        count = (data[col].isnull().sum()) / len(data)
        if count > 0.5:
            delCol.append(col)
            miss_col += 1

    data = data.drop(columns = delCol)
    # print(data)
    print('Because the missing rate is too high, delete',miss_col,'attribute columns with identical data values')


    # 删除值相同的属性列
    num1 = len(data.columns)
    data.drop(data.columns[data.std() == 0], axis=1, inplace=True)
    data = data.T.drop_duplicates().T
    num2 = len(data.columns)
    # print(data)
    print('Because of duplicate data, delete ',(num1 - num2),'attribute columns with identical data values')


    #众数填充
    for col in data.columns:
        x = data[col].mode()[0]                         #取每一列（每一属性）的众数，用于填充此列空值
        data[col].fillna(x,inplace = True)
    # print(data)
    # print("Fill in empty values with mode")
    # data.to_excel("nonom.xlsx")


    #归一处理（最大最小值标准化）
    for col in data.columns:
        d = data[col]
        MAX = d.max()
        MIN = d.min()
        data[col] = ((d - MIN) / (MAX - MIN))
    # print(data)

#字符串量化
def encoder(data):
    label = LabelEncoder()
    label.fit(data['K6'])  # 对字符串数据创建字典
    data['K6'] = label.transform(data['K6'])  # 寻找对应的量化数据
    return data


filepath = "TEST副本.xlsx"
data = pd.read_excel(filepath)
# 删除无效数据列（待定）16列
data = data.drop(['编号','K1', 'K2', 'K3', 'K4', 'K5', '所属联队', '姓名','记录日期','上高原日期', '填写人', '填写人所属连队', '职务', 'name33','wtime','name3'],axis=1)
encoder(data) #对来源地进行量化
newdata = data_processing(data)
newdata.to_excel("newdata.xlsx")



