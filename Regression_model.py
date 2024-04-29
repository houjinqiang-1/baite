import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score


def evaluation(y_test, y_predict):
    y_predict = np.array(y_predict)
    y_test = np.array(y_test)
    # mae = mean_absolute_error(y_test, y_predict)
    # mse = mean_squared_error(y_test, y_predict)
    # rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    # mape = (abs(y_predict - y_test) / y_test).mean()
    r_2 = r2_score(y_test, y_predict)
    smape = (2.0 * (abs(y_predict - y_test) / (abs(y_predict) + abs(y_test)))).mean()
    return smape,r_2  # smape


def sepTrainTestData(data, testIndex, usedColumns=None):
    if usedColumns != None:
        data = data[usedColumns]

    trainIndex = list(set(data.index) - set(testIndex))
    trainData = data.loc[trainIndex]
    testData = data.loc[testIndex]
    return trainData, testData

def model(data,usedCol):
    thisModelScore = []
    groupSize = 20

    # print(usedCol,len(usedCol))
    regr = linear_model.LinearRegression()
    for i in range(0, len(data), groupSize):
        if i + groupSize < len(data):
            testIndex = list(range(i, i + groupSize))
        else:
            testIndex = list(range(i, len(data)))
        # X:需要输入的数据 Y:需要得到的打分数据
        avg = ['t11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't31', 't32', 't33', 't34', 't35', 't36', 't37','t38']
        # realScores = data[avg].mean(axis = 1)
        dataY = data[avg].mean(axis=1)
        dataX = data.drop(columns = avg)

        # 分出训练集和测试集
        train_X, test_X = sepTrainTestData(dataX, testIndex=testIndex, usedColumns=usedCol)
        train_Y, test_Y = sepTrainTestData(dataY, testIndex=testIndex)

        regr.fit(train_X, train_Y)
        model_Y = regr.predict(test_X)
        # 将算出来的分数进行叠加
        thisModelScore += list(model_Y)
    # 调整score的格式
    # for i in range(len(thisModelScore)):
    #     thisModelScore[i] = thisModelScore[i][0]
    smape,r_2 = evaluation(dataY, thisModelScore)
    print('准确率=',1 - smape,'R_2',r_2)

def model2(data,usedCol):
    regr = linear_model.LinearRegression()

    # X:需要输入的数据 Y:需要得到的打分数据
    avg = ['t11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't31', 't32', 't33', 't34', 't35', 't36', 't37','t38']
    dataY = data[avg].mean(axis=1)
    # 分出训练集和测试集

    trainall_X = data[usedCol]
    trainall_Y = dataY

    #         print(train_X.shape,train_Y.shape)
    # 多元回归模型，进行预测
    regr.fit(trainall_X, trainall_Y)

    data_test = pd.read_excel("test_data.xlsx")
    test_X = data_test[usedCol]
    test_Y = data_test[avg].mean(axis = 1)

    model_Y = regr.predict(test_X)
    smape, r_2 = evaluation(test_Y, model_Y)
    print("准确率",1 - smape,"R2",r_2)



data = pd.read_excel("newdata.xlsx")
data = data.drop(data.columns[:1],axis = 1)#删除读文件时产生的unamed：0列
# print(data['T3_cfq'])
avg = ['t11','t12','t13','t14','t15','t16','t17','t18','t31','t32','t33','t34','t35','t36','t37','t38']
realScores = data[avg].mean(axis = 1)
#K_core
# useCol = ['D26','PCT_cq','PDW_cq','G36','J24','J25','t2C15','E6','G21','C19','D6']
# useCol = ['PDW_cq','T1_ES']
# useCol = ['T1_SP','T1_interference','T3_memory','PDW_cq','PCT_cq','G26','precise_2_mot_py','rt_change_120200_w01_shift_py','T1_ES','GS_cq','rate_hit_nwm_py']
# useCol = ['T1_SP', 'T1_ER', 'T1_interference', 'T3_memory', 'rough_2_4_mot_gy', 'rt_change_120200_w01_shift_gy', 'G26', 'acc_change_120200_w01_shift_gy', 'weight1', 'PDW_cq', 'PCT_cq']
# useCol = ['A3','B5','B12','B21','B25','B29','C2','C3','C4','C15','E3','E7','F4','F7','F8','G3','G6','G11','G12','G33','I13','J11','J18','K10','t2A10','t2A17','t2D5','t2D7','t2D12','t2D25','t2D40','t2E3','t2E17','t2E18','o1','o3','A3r','t3D17','t3D18','t3D23','t3E2','t3E3','t3E18','T3_OTP','T3_ORP','T2_rf','t1_OTP','t1_ORP','acc_change_120200_w01_shift_gy','SP_cq','HCT_sn1']
# useCol = ['A3','B5','B12','B21','B25','B29','C2','C3','C4','C15','E3','E7','F4','F7','F8','G3','G6','G11','G12','G33','I13','J11','J18','K10','o1','o3','A3r','t1_OTP','t1_ORP','acc_change_120200_w01_shift_gy','SP_cq','HCT_sn1']
# useCol = ['t2D6','t2E7']
# model2(data,useCol)
# thisModelScore = model(data,useCol)
# smape,r_2 = evaluation(realScores, thisModelScore)
# print('SMAPE=',1 - smape,'\n','R_2=',r_2)

model(data,useCol)