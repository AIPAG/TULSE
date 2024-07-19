import networkx as nx
import numpy as np
import pandas as pd
import random
import torch
import warnings

from collections import defaultdict
from gensim.models import Word2Vec
from sklearn import preprocessing
from tqdm import tqdm

from .showinfo import mytimer, func_with_timer

warnings.filterwarnings('ignore')

# v-1.1: 用一行代替了5行，这下好看多了。以及注意，pd.DataFrame的变量应该是一个指针，直接赋值给另一个变量并不会创建一个复制
# v-1.2: 删除了return，因为实际上就是返回一个指针，没有必要
def addLIDtoUserSubT(allTr:pd.DataFrame):
    allTr['lid'] = allTr[['rowID', 'colID']].astype(int).astype(str).agg('X'.join, axis=1)

# v-1.1: 同样是使用将DataFrame转化为list的方法优化时间复杂度。这中间利用了两个技巧：
#        1、使用defaultdict(lambda:init_value)将字典的初始值设置为init_value
#        2、使用pd.DataFrame.from_dict(data, orient="index")来将字典转化为DataFrame同时字典的key可以被设置为index
#        使用这种方法的一大好处是，由于我们创建dataframe时将lid作为了index，因此可以直接在保留lid信息的情况下对数据进行标准化
def timeslicing(subTrs:pd.DataFrame) -> pd.DataFrame:
    """计算时间频率向量，并对列做标准化

    Args:
        subTrs (pd.DataFrame): 轨迹数据集

    Returns:
        pd.DataFrame: 时间频率向量
    """
    columns_time = [f'hour_{i}' for i in range(24)]
    subTrs['time'] = pd.to_datetime(subTrs['utc'], utc=True, unit='s')
    subTrs['time'] = subTrs['time'].dt.hour
    subTrs.to_csv('./tmp/time.csv')

    with mytimer(desc='count time and frequency'):
        lids = subTrs['lid'].to_list()
        times = subTrs['time'].to_list()
        lid2times = defaultdict(lambda:[0]*24)
        # 时间计数
        for i in range(len(lids)):
            lid2times[lids[i]][times[i]] += 1
        ppde_matrix = pd.DataFrame.from_dict(lid2times, orient="index", columns=columns_time, dtype=float) # 字典转DataFrame
        ppde_matrix = ppde_matrix.sub(ppde_matrix.mean(axis=0), axis=1).div(ppde_matrix.std(axis=0), axis=1) # 标准化
        # ppde_matrix = ppde_matrix.sub(ppde_matrix.mean(axis=0), axis=1).div(ppde_matrix.std(axis=1), axis=0)
        ppde_matrix.reset_index(inplace=True)
        ppde_matrix.rename(columns={"index":"lid"}, inplace=True)

    subTrs.drop(columns=["time"],inplace=True)
    return ppde_matrix

# v-1.1: 同timeslicing，所以注释都懒得写了
def visitMatrix(subTrs:pd.DataFrame) -> pd.DataFrame:
    """计算空间频率向量

    Args:
        subTrs (pd.DataFrame): 轨迹数据集

    Returns:
        pd.DataFrame: 空间频率向量
    """
    num_user = max(subTrs['userID'].unique())+1
    columns_user = [f'user_{i}' for i in range(num_user)]

    with mytimer(desc='build visit matrix') as timer:
        users = subTrs['userID'].to_list()
        lids = subTrs['lid'].to_list()
        lid2users = defaultdict(lambda:[0]*num_user)
        for i in range(len(users)):
            lid2users[lids[i]][users[i]] += 1
        uce_matrix = pd.DataFrame.from_dict(lid2users, orient="index", columns=columns_user, dtype=float)
        uce_matrix = uce_matrix.sub(uce_matrix.mean(axis=0), axis=1).div(uce_matrix.std(axis=1), axis=0)
        uce_matrix.reset_index(inplace=True)
        uce_matrix.rename(columns={"index":"lid"}, inplace=True)

    return uce_matrix

# v-1.2: 原来baseline中使用到的函数，目前由于w2v中需要该函数因此移动到本文件中
def unpack_subtraj(data:pd.DataFrame, signs=('lid',), join=True):
    """将轨迹转化为子轨迹，并用字符串或字符串数组的形式作为返回值

    Args:
        data (pd.DataFrame): 轨迹数据集
        signs (tuple, optional): 需要提取的列，列的类型应该是或者可以转化为str. Defaults to ('lid',).
        join (bool, optional): 是否要拼接。True会以字符串形式返回，False则会以字符串数组形式返回. Defaults to True.

    Returns:
        _type_: 返回值第一个是uid的列表，后面返回signs中每个列组合成的子轨迹数组
    """
    data_group = data.groupby(['userID', 'TrID'], as_index=False)
    trajs = [[] for _ in signs]
    uids = []
    for (uid, _), traj in data_group:
        for i, sign in enumerate(signs):
            if join:
                trajs[i].append(' '.join(traj[sign].astype(str).values))
            else:
                trajs[i].append(traj[sign].astype(str).values.tolist())
        uids.append(uid)
    return uids, *trajs

def dataset_w2v(dataset:pd.DataFrame, embed_size:int) -> pd.DataFrame:
    """计算word2vec向量

    Args:
        dataset (pd.DataFrame): 轨迹数据集
        embed_size (int): w2v向量维度

    Returns:
        pd.DataFrame: w2v向量
    """
    _, trajs = unpack_subtraj(dataset, join=False)
    uids = dataset['lid'].unique()
    model = Word2Vec(trajs, vector_size=embed_size, window=5, min_count=1)
    columns = [f'w2v_{i}' for i in range(embed_size)]
    output = pd.DataFrame(model.wv[uids], columns=columns, dtype=float)
    ret = pd.concat((pd.DataFrame(uids, columns=['lid']), output), axis=1)
    return ret

###################################################
# 已经弃置的代码中使用到的函数
###################################################

@func_with_timer(desc='add lids to dataset')
def add_lid_step(dataset:pd.DataFrame, step:int) -> pd.DataFrame:
    for i in range(1, step+1):
        dataset[f'lid{i}'] = dataset[[f'rowID{i}', f'colID{i}']].astype(int).astype(str).agg('X'.join, axis=1)
        dataset.drop(columns=[f'rowID{i}', f'colID{i}'], inplace=True)
    return dataset

@func_with_timer("calculate and merge visit matrix and time slice with steps.")
def agg_step(dataset:pd.DataFrame, matches:list, multipler:float) -> pd.DataFrame:
    step = len(matches[0])
    columns = [f'lid{i}' for i in range(1, step+1)]
    step_weight = [1]
    for _ in range(step-1):
        step_weight.append(step_weight[-1]*multipler)
    weight_sum = sum(step_weight)
    for i in range(step):
        step_weight[i] /= weight_sum
    lids = dataset[columns].drop_duplicates().set_index(columns)
    for m in matches:
        mweighted:pd.DataFrame = sum(lids.join(m[i].set_index(f'lid{i+1}')*step_weight[i], on=f'lid{i+1}') for i in range(step))
        dataset = dataset.merge(mweighted, on=columns)
    return dataset

def embed_step(subTrs:pd.DataFrame, step:int, func, *args, **kwargs):
    ret = []
    for i in range(1, step+1):
        subi = subTrs.rename(columns={f'lid{i}':'lid'})
        ret.append(func(subi, *args, **kwargs).rename(columns={'lid':f'lid{i}'}))
    return ret

# lid直接onehot会让onehot维度太多。
# 两个列分别one-hot，然后拼接起来，即两个属性列，合并起来的标识中，会有两个位为1
# 这种方式会让onehot维度降低
# 并且将onehot作为多列存储。
def onehotEncoding3(allSubTr):
    tempdata = allSubTr[['rowID', 'colID']]
    # print(tempdata.head())
    # print("******************************")
    # tempdata1 = allSubTr['lid']
    # print(tempdata)
    # print(tempdata1)

    # 调用sklearn完成编码
    enc = preprocessing.OneHotEncoder()
    enc.fit(tempdata)  # enc已经训练好了

    # 构建一个新的DF，一列是lid，后面都是onehot0,onehot1......onehotn
    onehot = enc.transform(tempdata).toarray()

    onehot_size = len(onehot[0])

    # 构建一个新的DF，一列是lid，后面都是onehot0,onehot1......onehotn
    colnewname = ['onehot_' + str(i) for i in range(0, onehot_size)]
    newDF = pd.DataFrame(onehot, columns=colnewname)
    # onehot.rename(columns=dict(colnewname),inplace = True)
    # print(newDF)
    newDF['lid'] = allSubTr['lid']
    # print(newDF)
    # print(len(newDF))
    # print(newDF.dtypes)
    # newDF有7816行，说明有冗余重复的。因为网格的行是300多，列是500多，应该共800多个onehot
    newDF = newDF.drop_duplicates('lid')
    # print(len(newDF)) #1305

    return newDF, onehot_size


def onehotonData(allTr, tr, onehot_size):
    colnewname = ['onehot_' + str(i) for i in range(0, onehot_size)]
    dataheader = ['userID', 'TrID', "rowID", "colID", "utc", 'lid']
    newdataheader = dataheader + colnewname

    tr['userID'] = tr['userID'].astype('int16')
    tr['TrID'] = tr['TrID'].astype('int32')

    tr = pd.merge(tr, allTr, how='left', on=['lid'])
    return tr


def approximateOnehotEmbed(allTr, trainSet, testSet):
    allTr, onehotSize = onehotEncoding3(allTr)
    trainSetinOnehot = onehotonData(allTr, trainSet, onehotSize)
    testSetinOnehot = onehotonData(allTr, testSet, onehotSize)
    return trainSetinOnehot, testSetinOnehot, onehotSize

def deepwalkEmbedding(subTrs,trainSet, testSet,vector_size):
    allSubT = subTrs

    group = allSubT.groupby(["userID", "TrID"])
    allDF = pd.DataFrame()
    for g in group:
        df = g[1]['lid']
        array = df.values
        array.tolist()
        tmpList = []
        if len(array) == 1:
            edge = array[0] + ' ' + array[0]
            tmpList.append(edge)
        else:
            for i in range(1, len(array)):
                edge = array[i - 1] + ' ' + array[i]
                tmpList.append(edge)
        allDF = allDF.append(tmpList, ignore_index=True)
    allDF.to_csv("graph.csv", index=False, header=False)

    graph = nx.read_edgelist('graph.csv', create_using=nx.DiGraph())
    # 得到所有节点
    nodes = list(graph.nodes())
    # 得到序列
    walks = _simulate_walks(nodes, num_walks=10, walk_length=30, g=graph)
    w2v_model = Word2Vec(walks, sg=1, hs=1, vector_size=vector_size)
    model = w2v_model.wv

    w2v_data = model[subTrs['lid']]
    colnewname = ['w2v_' + str(i) for i in range(0, vector_size)]
    newDF = pd.DataFrame(w2v_data, columns=colnewname)
    #newDF['lid'] = allSubT['lid']
    rst = pd.concat([allSubT['lid'],newDF],axis=1)
    return rst


def deepwalk_walk(walk_length, start_node, G):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(random.choice(cur_nbrs))
        else:
            break
    return walk


# 产生随机游走序列
def _simulate_walks(nodes, num_walks, walk_length, g):
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for v in nodes:
            walks.append(deepwalk_walk(walk_length=walk_length, start_node=v, G=g))
    return walks


def richEmbed(allTr, trainSet, testSet):
    allTr, onehotSize = onehotEncoding3(allTr)
    trainSetinOnehot = onehotonData(allTr, trainSet, onehotSize)
    testSetinOnehot = onehotonData(allTr, testSet, onehotSize)
    return trainSetinOnehot, testSetinOnehot, onehotSize
