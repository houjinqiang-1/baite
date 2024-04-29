import networkx as nx
import embedding as emb
import pandas as pd
import matplotlib.pyplot as plt

# 计算各属性之间得Pearson相关系数
# data - 处理完之后的数据表
# ifabs - 相关系数计算中是否保留负相关系数
# return 得到的相关系数矩阵
def calPearson(data, ifabs):
    import numpy as np
    matrix = np.zeros((len(data.columns), len(data.columns)), )          #创建相关系数矩阵
    coor = pd.DataFrame(matrix, index=data.columns, columns=data.columns)

    for coli in data.columns:
        for colj in data.columns:
            try:
                r = np.corrcoef(data[coli], data[colj])
                resCoor = r[0][1]
                if ifabs == True:
                    resCoor = abs(resCoor)
                coor[coli][colj] = resCoor
            except:
                continue

    return coor

# 将相关系数大小作为节点间权重建网
def Connect(resCoor):
    dic1 = emb.generate_dic(matrix=resCoor, col=resCoor.columns)  # 这里的resCoor是相关度矩阵
    dic1 = emb.sort_value(old_dict=dic1, reverse=True)

    G = nx.Graph()
    G.add_nodes_from(resCoor.columns)

    # stopPoint = 100000
    # Graph, GCC, SGCC = emb.stopBy(Graph=G, newdic=dic1, stopPoint=stopPoint)
    Graph, GCC ,SGCC, stopPoint = emb.stopBy2(Graph=G, newdic=dic1)
    # print(stopPoint)
    emb.plot_connect1(x=list(range(stopPoint)), arr1=GCC, arr2=SGCC)

    # 确认建网点
    stoppoint = stopPoint
    graphPath = "graph" + str(stoppoint) + ".graphml"
    nx.write_graphml_lxml(Graph, graphPath)
    # drawDegree_distribution(Graph) #网络度分布图
    return Graph

def drawDegree_distribution(G):

    degree = nx.degree_histogram(G)  # 返回图中所有节点的度分布序列

    x = range(len(degree))  # 生成X轴序列，从1到最大度
    y = [z / float(sum(degree)) for z in degree]  # 将频次转化为频率
    plt.figure(figsize=(5.8, 5.2), dpi=150)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
    plt.xlabel("Degree", size=14)  # Degree
    plt.ylabel("Frequency", size=14)  # Frequency
    plt.xticks(fontproperties='Times New Roman', size=13)
    plt.yticks(fontproperties='Times New Roman', size=13)
    plt.loglog(x, y, '.')
    plt.show()  # 显示图表

def drawcorrelation(G):
    plt.hist(G, bins=100)

    plt.title("correlation analyze")
    plt.xlabel("corrcoef")
    plt.ylabel("distribution")
    plt.show()


# data = pd.read_excel('newdata.xlsx')
# print(data)
# data = data.drop(data.columns[:1],axis = 1)#删除读文件时产生的unamed：0列
# print(data)
# # 删除他人评价列
# data = data.drop(['t11','t12','t13','t14','t15','t16','t17','t18','t31','t32','t33','t34','t35','t36','t37','t38'],axis = 1)
# print(data)
#
# resCoor = calPearson(data, ifabs=False)#相关系数矩阵
#
# # drawcorrelation(resCoor)#绘制相关性分布
#
# # resCoor.to_excel('newdata_Pearson.xlsx')
# # print(resCoor)
#
# graph = Connect(resCoor)#建网
#
# # #绘制建网处度分布
# # # G = nx.read_graphml("graph1200.graphml")
# # # drawDegree_distribution(G)

