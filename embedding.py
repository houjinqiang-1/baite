"""
为了简化代码结构

"""

import pandas as pd 
#from termcolor import colored
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
def swap(t1, t2):
    return t2, t1

def drawDistributionOfMatrix(Matrix,gapNum):
    """
    计算matrix里面的元素分布
    """
    import numpy as np
    left = np.amin(Matrix)
    right = np.amax(Matrix)
    import matplotlib.pyplot as plt 
#     gapSize = 0.01
#     left = -1
#     right = 1
    if left>right:
        swap(left,right)
    gapSize = (right - left)/gapNum
#     gapNum = int((right - left)/gapSize)+1 
    # counts = np.zeros(gapNum+5)
    
    X = []
    for i in range(gapNum):
        num = gapSize * i + left
        num = round(num, 2)
        X.append(num)

    counts = np.zeros(gapNum)
    for i in range(len(Matrix)):
        for j in range(len(Matrix[i])):
            loc = (Matrix[i][j] - left)/gapSize
            loc -= 1
            loc = int(loc)
    #         print(loc)
            counts[loc] += 1 

    plt.scatter(X,counts,s=5)
    plt.xlabel("Value")
    plt.ylabel("The corresponding NUMBER of this value")
    
    sumCount = 0
    for i in counts:
        sumCount += i 
    print(sumCount,Matrix.shape)
    print("left == ",left,"/ right == ",right,"/ gapNum == ",gapNum,"/ gapSize == ",gapSize)
    
def show_bulidings_simply(buildings):
    X = []
    Y = []
    for i in range(len(buildings)):
        loc = np.float_(buildings[i].split(",")) 
        X.append(loc[0])
        Y.append(loc[1])
    plt.scatter(X,Y)

def drawGCCandSGCC(G,GCC,SGCC,pos,GCCcolor = "red",GCCnodeSize = 20 ,SGCCcolor = "green",SGCCnodeSize = 20 , othersColor = "blue" ,othersNodeSize = 5):
    """
    用来画跳变前后的graph（当然不是跳变前后也行）
   G,需要做图的G
   GCC,G的最大连通子图
   SGCC,G的次大连通子图
   pos，G的layout
    
    """
    colors = []
    sizes = []


    for i in range(len((G.nodes))):
        if i in GCC:
            colors.append(GCCcolor)
            sizes.append(GCCnodeSize)
        elif i in SGCC:
            colors.append(SGCCcolor)
            sizes.append(SGCCnodeSize)
        else:
            colors.append(othersColor)
            sizes.append(othersNodeSize)

#     len(colors)==len(sizes)
    nx.draw(G,pos=pos,node_size = sizes,node_color = colors)
    
def generate_dic(matrix,col):
    """
    input: 
        matrix:边权 dataframe类型
        col:matrix的列名 list类型 
        
    output:
        locToVal :将位置信息转化为权值 
    """
    locToVal = OrderedDict()
    for i in range(0,len(col)):
#         if i == len(col)-2:
#             continue #不然j的访问就出界啦！
        if i == len(col)-1:
            continue
        for j in range(i+1,len(col)):
#             print(i,len(col),j)
            val = matrix[col[i]][col[j]]
            if math.isnan(val):
                continue
                     
#             if val < 0: #不考虑负相关
#                 continue
#             print(val,type(val)," ",j)
            loc = (col[i],col[j])
            
            locToVal[loc] = val 
    return locToVal



# 普通 dict 插入元素时是无序的，使用 OrderedDict 按元素插入顺序排序
# 对字典按key排序, 默认升序, 返回 OrderedDict
from collections import OrderedDict

def sort_key(old_dict, reverse=False):
    """对字典按key排序, 默认升序, 不修改原先字典"""
    # 先获得排序后的key列表
    keys = sorted(old_dict.keys(), reverse=reverse)
    # 创建一个新的空字典
    new_dict = OrderedDict()
    # 遍历 key 列表
    for key in keys:
        new_dict[key] = old_dict[key]
    return new_dict


# 对字典按 value 排序，默认升序, 返回 OrderedDict
def sort_value(old_dict, reverse=False):
    """对字典按 value 排序, 默认升序, 不修改原先字典"""
    # 获取按 value 排序后的元组列表
    items = sorted(old_dict.items(), key=lambda obj: obj[1], reverse=reverse)
    # 创建一个新的空字典
    new_dict = OrderedDict()
    # 遍历 items 列表
    for item in items:
        # item[0] 存的是 key 值
        new_dict[item[0]] = old_dict[item[0]]
    return new_dict

def plot_connect(x,arr1,arr2):
    # plot
    labels = ['Gc','Sub_Gc']
#     x = [i/len(arr1) for i in range(len(arr1))]
    fig, ax_f = plt.subplots(figsize=(8, 7))
    ax_c = ax_f.twinx()
    f, = ax_f.plot(x,arr1,color='deeppink',linestyle='-', linewidth=1, marker='o',  markersize=2, label = labels[0])
#     ax_f.set_xlim(0, 1)
    c, = ax_c.plot(x,arr2,color='darkblue',linestyle='-', linewidth=1, marker='o',  markersize=2, label = labels[1])
    ax_f.set_ylim(0,max(arr1)+10)
    ax_c.set_ylim(0,max(arr1)+10)
    ax_f.legend(handles=[f,c],loc='upper right',fontsize=14)
    ax_f.tick_params(labelsize=18)
    ax_c.tick_params(labelsize=18)
    plt.show()

def plot_connect1(x,arr1,arr2):
    # plot
    labels = ['Gc','Sub_Gc']
#     x = [i/len(arr1) for i in range(len(arr1))]
    fig, ax_f = plt.subplots(figsize=(8, 7))
    ax_c = ax_f.twinx()
    f, = ax_f.plot(x,arr1,color='deeppink',linestyle='-', linewidth=1, marker='o',  markersize=2, label = labels[0])
#     ax_f.set_xlim(0, 1)
    c, = ax_c.plot(x,arr2,color='darkblue',linestyle='-', linewidth=1, marker='o',  markersize=2, label = labels[1])
    ax_f.set_ylim(0,max(arr1)+0.1)
    ax_c.set_ylim(0,max(arr2)+0.1)
    ax_f.legend(handles=[f,c],loc='upper right',fontsize=14)
    ax_f.tick_params(labelsize=18)
    ax_c.tick_params(labelsize=18)
    plt.show()
    fig.savefig("网络相变图.jpg")
def show_GCC_and_SGCC(GCC,SGCC,size,left,right):
    GCC1 = np.array(GCC)/size
    SGCC1 = np.array(SGCC)/size
    x = list(range(len(GCC)))
    
    plot_connect1(x[left:right],GCC1[left:right],SGCC1[left:right])

def stopBy_returnGraphs(Graph,newdic,stopPoint):
    """
    dic:构建graph的字典
    stopPoint:加到多少条边时停下

    返回：停下时对应的Graph以及graph
    """

    i = 0
    for key,val in newdic.items():#按照weight 从大到小的顺序加边
        if Graph.has_edge(key[0],key[1]) == True  or Graph.has_edge(key[1],key[0]) == True:
            continue
        Graph.add_edge(key[0],key[1],weight = val)
        i += 1
        if i >= stopPoint:
            Gcc = sorted(nx.connected_components(Graph), key=len, reverse=True)         
            return Graph,Gcc
        
def stopBy(Graph,newdic,stopPoint):# 输出图表，手动选点建网
    """
    dic:构建graph的字典
    stopPoint:加到多少条边时停下
    
    返回：停下时对应的Graph
    """
    GCC = []
    SGCC = []
    i = 0
    for key,val in newdic.items():#按照weight 从大到小的顺序加边
        Graph.add_edge(key[0],key[1])
        
    #     nx.draw(G)
        Gcc = sorted(nx.connected_components(Graph), key=len, reverse=True)
#         print(i,len(Gcc))
        G0 = Graph.subgraph(Gcc[0])
        GCC.append(len(G0.nodes()))
        
        if len(Gcc) == 1:
            SGCC.append(0)
        else:
            G1 = Graph.subgraph(Gcc[1])
            SGCC.append(len(G1.nodes()))
        i += 1
        if i >= stopPoint:
            return Graph,GCC,SGCC


def stopBy2(Graph, newdic):# 自动在跳变点建网
    """
    dic:构建graph的字典
    stopPoint:加到多少条边时停下

    返回：停下时对应的Graph
    """
    GCC = []
    SGCC = []
    max = 0
    i = 0
    for key, val in newdic.items():  # 按照weight 从大到小的顺序加边
        Graph.add_edge(key[0], key[1])
        node = len(Graph.nodes())

        #     nx.draw(G)
        Gcc = sorted(nx.connected_components(Graph), key=len, reverse=True)
        #         print(i,len(Gcc))
        G0 = Graph.subgraph(Gcc[0])
        GCC.append(len(G0.nodes()))
        if len(Gcc) == 1:
            SGCC.append(0)
            G1len = 0  # 第二联通团节点数
        else:
            G1 = Graph.subgraph(Gcc[1])
            SGCC.append(len(G1.nodes()))
            G1len = len(Graph.subgraph(Gcc[1]).nodes())  # 第二联通团节点数

        G0len = len(Graph.subgraph(Gcc[0]).nodes())

        if (G1len < max) & (G0len > 0.8*node):
            stopPoint = i + 1
            return Graph, GCC, SGCC, stopPoint
        else:
            max = G1len
            i += 1


def draw_origin_graph(G):
    # draw graph
    pos = nx.random_layout(G)  # compute graph layout
    plt.figure(figsize=(8, 8))  # image is 8 x 8 inches
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=100, cmap=plt.cm.RdYlBu, node_color='deeppink',alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show(G)

def countDictionaryValue(dic):
    """
    dic的value代表社团的id 
    统计每个社团到底有多少个，以及每个社团的元素都是谁 
    
    input：dic 字典 列名：社团id 
    output:
        counts: 字典类型 社团id：社团size（结点个数） 
        member ：字典类型 社团id：社团成员（list） 
    """
    counts = dict()
    member = dict() 
    for key,val in dic.items():
        try:
            counts[val] += 1
            member[val].append(key)
        except:
            counts[val] = 1
            member[val] = []
            member[val].append(key)
            
    return counts,member


def h_index(G, u):
    '''
    @description: Calculate the h-index of node u.
    @param : Graph G, node u
    @return: h-index of node u
    '''    
    # Define initial h_index equals 0.
    hi = 0                     
    # Define node u's neighbors.
    ns = {n:G.degree[n] for n in G[u]}                     
    # Sorted the neighbors in increasing order with node degree.
    sns = sorted(zip(ns.keys(), ns.values()), key=lambda x:x[1], reverse=True)
    # print(sns)
    for i, n in enumerate(sns):
        if i >= n[1]:
            hi = i
            break
        hi = i+1
    return hi

def k_shell(G, u=None):
    '''
    @description: Calculate the k-core of node u.
    @param : 
    @return: 
    '''
    _G = G.copy()
    # print(nx.info(_G))
    data = {}
    ks = 1
    while _G.nodes():
        # 暂存度为ks的顶点
        temp = []
        ns = {n:_G.degree[n] for n in _G.nodes()}
        # 每次删除度值最小的节点而不能删除度为ks的节点否则产生死循环。这也是这个算法存在的问题。
        kks = min(ns.values())
        while True:
            for k, v in ns.items():
                if v == kks:
                    temp.append(k)
                    _G.remove_node(k)
            ns = {n:_G.degree[n] for n in _G.nodes()}
            if kks not in ns.values():
                break
        data[ks] = temp
        ks += 1
    # If calcualte node u's k-core, return a single value.
    if u or u == 0:
        kc = 0
        for k,v in data.items():
            if u in v:
                kc = k
                break
        return kc    
    return data



def draw_origin_graph(G):
    # draw graph
    pos = nx.spring_layout(G)  # compute graph layout
    plt.figure(figsize=(8, 8))  # image is 8 x 8 inches
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=100, cmap=plt.cm.RdYlBu, node_color='deeppink',alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show(G)
    
    

    
def cal_weight_edges_gradual(dis):
    edges = []
    for i in range(len(dis)):
        for j in range(i + 1, len(dis)):
            edges.append((i,j,dis[i][j]))
    edges_sorted = sorted(edges,key=lambda x:(x[2]),reverse=True)
    return edges_sorted

def generate_noweight_gml_gradual(distance):
    nodes_list = [i for i in range(len(distance))]
    gnode = len(distance)
    G = nx.MultiGraph()
    
    # 按照权重大小排列的边
    edges = cal_weight_edges_gradual(distance)
    
    Gc_list = []
    subGc_list = []
    x_list = []
    data_name  = ''
    for t in range(len(edges)):
        G.add_nodes_from([edges[t][0],edges[t][1]])
        G.add_edge(edges[t][0],edges[t][1])
        for i in G.nodes():
            G.nodes[i]['value'] = str(i)
            G.nodes[i]['label'] = str(i)
#         print('Graph : node %d edges %d'%(G.number_of_nodes(), G.number_of_edges()))
        
        # 第二大连通子图
        count = 0
        connected = list(nx.connected_components(G))
        subgraphs = [G.subgraph(i) for i in connected]