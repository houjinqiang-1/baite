# encoding: utf-8

import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt

def visGraphPartition(G):
    """
    可视化G通过louvain算法划分的不同社团，并标上不同颜色
    """
    import colorsys
    import random

    def get_n_hls_colors(num):
        hls_colors = []
        i = 0
        step = 360.0 / num
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
            _hlsc = [h / 360.0, l / 100.0, s / 100.0]
            hls_colors.append(_hlsc)
            i += step

        return hls_colors

    def ncolors(num):
        rgb_colors = []
        if num < 1:
            return rgb_colors
        hls_colors = get_n_hls_colors(num)
        for hlsc in hls_colors:
            _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
            r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
            rgb_colors.append([r, g, b])

        return rgb_colors

    def RGB_to_Hex(RGB):
        color = '#'
        for i in RGB:
            num = int(i)
            # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
            color += str(hex(num))[-2:].replace('x', '0').upper()
        return color

    comms = nx_comm.louvain_communities(G, seed=123)

    pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    options = {"edgecolors": "tab:gray", "node_size": 80, "alpha": 0.9}
    temp = ncolors(len(comms))
    for index, x in enumerate(comms):
        color = RGB_to_Hex(temp[index])
        nx.draw_networkx_nodes(G, pos, nodelist=list(x), node_color=color, **options)
    # edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # print("number of nodes: ", len(G.nodes()))
    plt.title("one color one community")
    plt.tight_layout()
    plt.axis("off")
    plt.show()
    return

# 根据对应值进行排序，并返回每个社团排序后前windowSize个值
# 先按measureDict1排序，然后按measureDict2排序
def calTopCenters(part, measureDict1 ,measureDict2, windowSize):
    # get ranked partNode:measurement
    tempDict = {}
    for node in part:
        temp = [0, 0]
        temp[0] = measureDict1[node]
        temp[1] = measureDict2[node]
        tempDict[node] = temp
    tempDict = dict(sorted(tempDict.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True))
    # print(tempDict)
    #     # print(type(tempDict))
    # print(tempDict)
    if windowSize > len(part):
        return list(tempDict.keys())

    resNode = []
    i = 0
    for key in tempDict.keys():
        i += 1
        if i > windowSize:
            break
        resNode.append(key)
    return resNode


def calCommunityCenter(G, partition, chooseRule, windowSize):
    """
    algrithm for choossing community center:
        kcore, degree, hindex

    @parameter:
        input:
            G:graph (networkx)
            partition
            windowSize
        output:
            centers:list of list. every element in the list shows the length of partition and centers of corresponding every communities.
    """
    #返回各节点k_shell值
    #返回各节点与其所在K-core
    def calKcore(G):
        return nx.core_number(G)
    #返回各节点度数
    def calDegree(G):
        return nx.degree(G)
    #返回H指数（有i个邻居的度数不小于i，但是没有i+1个邻居的度数不小于i+1）
    def calHindex(G):
        def hindex(g, n):
            nd = {}
            h = 0
            for v in g.neighbors(n):
                nd[v] = g.degree(v)
                snd = sorted(nd.values(), reverse=True)
                for i in range(0, len(snd)):
                    h = i
                    if snd[i] < i:
                        break
            return h

        hindexDict = dict()
        for node in G.nodes():
            hindexDict[node] = hindex(G, node)
        return hindexDict

    if chooseRule == "kcore":
        measureDict1 = calKcore(G)  # return dictionary which means: {node:kcore of this node}
        measureDict2 = calDegree(G)
    elif chooseRule == "degree":
        measureDict1 = calDegree(G)
        measureDict2 = calDegree(G)
    else:
        measureDict1 = calHindex(G)
        measureDict2 = calDegree(G)

    resTopCenters = []
    for par in partition:
        nodes = calTopCenters(part = par, measureDict1 = measureDict1, measureDict2 = measureDict2, windowSize = windowSize)
        resTopCenters.append([len(par), nodes])
    return resTopCenters


def showCommunityCenters(communityCenter, graphName, windowSize, NotShow1=True):
    # print("*" * 10, graphName, " 前{}个点".format(windowSize), "*" * 10)
    cri = []
    for center in communityCenter:
        if center[0] == 1 and NotShow1:
            continue
        # print("partition size ", center[0], " center ", center[1])
        if center[0] > 9 :
            cri.append(center[1][0])
    return cri



Rules = ["kcore", "degree", "hindex"]#选取社团中心时的规则选择，将选0，即将K-shell值作为第一因素，度值作为第二因素。选2，即将H-index值作为第一因素，度值作为第二因素

ConnectedG = nx.read_graphml("graph9018.graphml")#获取所建网络

partition109fullConnect = nx_comm.louvain_communities(ConnectedG, seed=123)
print(partition109fullConnect)

windowSize = 20   #每个社团中最关键的前windowSize个指标
ruleIndex = 0   #规则选择

print("rule : ",Rules[ruleIndex])


communityCenterTemp = calCommunityCenter(G = ConnectedG,partition = partition109fullConnect,windowSize = windowSize,chooseRule = Rules[ruleIndex])
cri = showCommunityCenters(communityCenterTemp," model 9100 full connected centers ",windowSize ,True)   #打印社团中心
print(cri)
visGraphPartition(ConnectedG)

