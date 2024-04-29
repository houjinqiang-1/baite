import pandas as pd
import networkx.algorithms.community as nx_comm
import data_processing as dp
import network_building as nb
import Community_division as Cd
import Regression_model as Rm

filepath = r"C:\Users\凉丶凉丶\Desktop\基于复杂网络视角的JJGY人员选拔研究\TEST副本.xlsx"
data = pd.read_excel(filepath)

## 数据预处理
# 删除无效数据列（待定）16列
data = data.drop(['编号','K1', 'K2', 'K3', 'K4', 'K5', '所属联队', '姓名','记录日期','上高原日期', '填写人', '填写人所属连队', '职务', 'name33','wtime','name3'],axis=1)
dp.encoder(data) #对来源地进行量化
newdata = dp.data_processing(data)
newdata.to_excel('newdata.xlsx')

## 相关关系网络构建
data = pd.read_excel('newdata.xlsx')
data = data.drop(data.columns[:1],axis = 1)#删除读文件时产生的unamed：0列
# 删除他人评价列
data = data.drop(['t11','t12','t13','t14','t15','t16','t17','t18','t31','t32','t33','t34','t35','t36','t37','t38'],axis = 1)

resCoor = nb.calPearson(data, ifabs=False)#相关系数矩阵

graph = nb.Connect(resCoor)#建网

## 社团划分
Rules = ["kcore", "degree", "hindex"]#选取社团中心时的规则选择，将选0，即将K-shell值作为第一因素，度值作为第二因素。选2，即将H-index值作为第一因素，度值作为第二因素

ConnectedG = graph#获取所建网络

partition109fullConnect = nx_comm.louvain_communities(ConnectedG, seed=123)
# print(partition109fullConnect)

windowSize = 20   #每个社团中最关键的前windowSize个指标
ruleIndex = 0   #规则选择

# print("rule : ",Rules[ruleIndex])

communityCenterTemp = Cd.calCommunityCenter(G = ConnectedG,partition = partition109fullConnect,windowSize = windowSize,chooseRule = Rules[ruleIndex])
useCol = Cd.showCommunityCenters(communityCenterTemp," model 9100 full connected centers ",windowSize ,True)   #打印社团中心
print(useCol)
# useCol = ['T1_SP','T1_interference','T3_memory','PDW_cq','PCT_cq','G26','precise_2_mot_py','rt_change_120200_w01_shift_py','T1_ES','GS_cq','rate_hit_nwm_py']
# print("Perform",int(len(data)/windowSize) + 1,"tests and take the average")
Cd.visGraphPartition(ConnectedG)

## 模型预测
data = pd.read_excel("newdata.xlsx")

avg = ['t11','t12','t13','t14','t15','t16','t17','t18','t31','t32','t33','t34','t35','t36','t37','t38']
realScores = data[avg].mean(axis = 1)
Rm.model(data,useCol)
