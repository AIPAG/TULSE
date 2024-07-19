import math
import random
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from math import sqrt
from random import gauss
from tqdm import tqdm

from .loaddataset import rename_uid
from .showinfo import func_with_timer, show_dataset_info

# v-1.2: 删除了return，因为没必要
def coordinate2grid(allCheckins:pd.DataFrame, cellSize: float):
    """给轨迹数据集划分网格

    Args:
        allCheckins (pd.DataFrame): 轨迹数据集\n
        cellSize (float): 网格长度，单位为km。实际长度与该数值有一定的偏差，但不超过1%，因此可以视为准确数值
    """
    maxlon, minlon = allCheckins["longitude"].max(), allCheckins["longitude"].min()
    maxlat, minlat = allCheckins["latitude"].max(), allCheckins["latitude"].min()
    midlat = (maxlat + minlat) / 2
    km_per_lon, km_per_lat = 111.32 * math.cos(math.radians(midlat)), 111
    deg_lon, deg_lat = cellSize / km_per_lon, cellSize / km_per_lat
    lon_grids, lat_grids = int((maxlon - minlon) / deg_lon) + 1, int((maxlat - minlat) // deg_lat) + 1
    grid_lon, grid_lat = (maxlon - minlon) / lon_grids, (maxlat - minlat) / lat_grids
    allCheckins["longitude"] = allCheckins["longitude"].sub(minlon).floordiv(grid_lon).astype(int)
    allCheckins["latitude"] = allCheckins["latitude"].sub(minlat).floordiv(grid_lat).astype(int)
    allCheckins.rename(columns={"longitude": "colID", "latitude": "rowID"}, inplace=True)

# v-1.1: 改变了原本的效率无比低下的串行遍历DataFrame的方法，而是转化为了列表进行计算
#        同时，添加了防拥挤机制，在一定时间内一直处于同一个区域的多个点将被压缩
@func_with_timer("spliting checkins into subtrajs")
def checkins2subtraj(checkins:pd.DataFrame, interval_hour:int, a:int, step:int=0) -> int:
    """根据间隔划分子轨迹，同时在一个区域内的连续多个点将被压缩。

    Args:
        checkins (pd.DataFrame): 轨迹数据集\n
        interval_hour (int): 时间间隔，单位为小时\n
        a (int): 压缩算法底数，具体压缩算法为：第x个点与第1个点的时间差至少为a^x秒\n
        step (int, optional): depricated. Defaults to 0.

    Returns:
        _type_: 切割后的轨迹，与子轨迹数目
    """
    f = open("./log/geolife_subtraj.log", 'w')
    interval = interval_hour * 3600
    checkins.sort_values(by=["userID", "utc"], inplace=True, ignore_index=True)
    subtrajID, subtrajlen, rmnum = 0, 1, 0
    trajnum = 1
    users, utcs = checkins["userID"].to_list(), checkins["utc"].to_list()
    rows, cols = checkins["rowID"].to_list(), checkins["colID"].to_list()
    pi = 0 # pi表示在当前grid出现的点的最早的点的序号
    trID = [0]*len(checkins)
    # bcut, acut = [], []
    output = "the subtraj {} for user {} has {} checkins, and cut to {} checkins\n"
    def adjust(pi:int, ti:int, a:int) -> int:
        t = a
        utc = utcs[pi]
        remove_num = 0
        for i in range(pi+1, ti):
            if utcs[i]-utc>=t:
                # t *= a
                utc = utcs[i]
            else:
                trID[i] = -1
                remove_num += 1
        return remove_num
    for i in range(1, len(checkins)):
        if users[i]!=users[pi]:
            rmnum += adjust(pi, i, a)
            tout = output.format(subtrajID, users[pi], subtrajlen, subtrajlen-rmnum)
            # bcut.append(subtrajlen)
            # acut.append(subtrajlen-rmnum)
            f.write(tout)
            subtrajID, subtrajlen, rmnum = 0, 1, 0
            trajnum += 1
            pi = i
        elif utcs[i]-utcs[i-1]>=interval:
            rmnum += adjust(pi, i, a)
            tout = output.format(subtrajID, users[pi], subtrajlen, subtrajlen-rmnum)
            # bcut.append(subtrajlen)
            # acut.append(subtrajlen-rmnum)
            f.write(tout)
            subtrajID += 1
            trajnum += 1
            subtrajlen, rmnum = 1, 0
            pi = i
        elif rows[i]!=rows[pi] or cols[i]!=cols[pi]:
            rmnum += adjust(pi, i, a)
            subtrajlen += 1
            pi = i
        else:
            subtrajlen += 1
        trID[i] = subtrajID
    tout = output.format(subtrajID, users[pi], subtrajlen, subtrajlen-rmnum)
    # bcut.append(subtrajlen)
    # acut.append(subtrajlen-rmnum)
    # print(tout)
    f.write(tout)
    checkins.insert(1, "TrID", trID)
    checkins = checkins[checkins["TrID"]!=-1]
    columns = ['userID', 'TrID', 'rowID', 'colID', 'utc']
    for i in range(1, step+1):
        columns.append(f'rowID{i}')
        columns.append(f'colID{i}')
    checkins = checkins[columns]
    checkins.reset_index(inplace=True, drop=True)
    f.write(f"\ntotally {len(checkins)} checkins.")
    f.close()
    # plt.title("dataset cut")
    # plt.scatter(bcut, acut, marker='.', s=1.5)
    # plt.plot([0, 500], [0, 500], linewidth=0.33)
    # plt.xlabel("length before cut")
    # plt.ylabel("length after cut")
    # plt.show()
    return checkins, trajnum

# v-1.1: 新添加功能，实际上是从自己以前的代码复制粘贴过来的（
@show_dataset_info(["dataset with all checkin"])
def filt_trajnum(dataset:pd.DataFrame, trajnum:int):
    """筛选至少拥有特定数目子轨迹的用户

    Args:
        dataset (pd.DataFrame): 轨迹数据集，其中子轨迹应该从1开始编号\n
        trajnum (int): 用户至少拥有多少条子轨迹

    Returns:
        _type_: 筛选后的数据集，和子轨迹数量
    """
    ''' 筛选出至少拥有threshold条子轨迹的用户 '''
    counter = dataset.groupby("userID", as_index=False)["TrID"].max()
    counter = counter[counter["TrID"]>=trajnum]
    uids = counter['userID']
    dataset = dataset[dataset['userID'].isin(uids)]
    rename_uid(dataset)
    dataset.reset_index(inplace=True, drop=True)
    print(f"num of user after trajcut: {len(uids)}")
    return dataset, sum(counter['TrID'])
                

# v-1.1: 用groupby的sample函数自动采样代替了串行shuffle然后采样的方式，将效率提高了10倍左右，同时sample函数同时支持按个数与按比例采样，可扩展性高
@func_with_timer('spliting data into train and test')
@show_dataset_info(['train dataset', 'test dataset'])
def splitData(allTrajectoryData:pd.DataFrame, testNum:int):
    """将数据集分为测试集与训练集

    Args:
        allTrajectoryData (pd.DataFrame): 轨迹数据集
        testNum (int): 每个用户在测试集中的轨迹数

    Returns:
        _type_: 测试集，训练集，测试集轨迹数目，训练集轨迹数目
    """
    total_list = allTrajectoryData[['userID', 'TrID']].drop_duplicates()
    # test_list = total_list.groupby("userID").sample(frac=0.2)
    test_list = total_list.groupby("userID").sample(n=testNum)
    dataset = allTrajectoryData.merge(test_list, indicator=True, how='left')
    dataset["filter"] = dataset["_merge"]=="both"
    dataset.drop(["_merge"], axis=1, inplace=True)
    train_data, test_data = dataset[dataset["filter"]==False], dataset[dataset["filter"]==True]
    train_data.drop(["filter"], axis=1, inplace=True)
    test_data.drop(["filter"], axis=1, inplace=True)
    return train_data, test_data, len(total_list)-len(test_list), len(test_list)

# v-1.1: 让代码更短了，其实效率提升不到一倍吧（
@func_with_timer("DataFrame to Tensor")
def toTenser(data:pd.DataFrame, step:int=0):
    """将轨迹按子轨迹转化为Tensor

    Args:
        data (pd.DataFrame): 轨迹数据集
        step (int, optional): depricated. Defaults to 0.

    Returns:
        _type_: 轨迹Tensor组成的集合
    """
    Tr_after_group = data.groupby(['userID', 'TrID'], as_index=False)
    # 这个时候Tr_after_group作为一个迭代器，其返回值为(类别，类别内的值)组成的二元组
    if step==0:
        drop_columns = ['TrID', 'rowID', 'colID', 'utc', 'lid']
    else:
        drop_columns = ['TrID', 'utc']+[f'lid{i}' for i in range(1, step+1)]
    return [torch.Tensor(group.drop(columns=drop_columns).values) for _, group in Tr_after_group]

# v-1.2: ?
@func_with_timer("trans trajs to vectors and graphs")
def traj2graph(dataset:pd.DataFrame, rlen:int, clen:int, rmin:int, cmin:int, multipler:int):
    trajs = dataset.groupby(['userID', 'TrID'], as_index=False)
    drop_columns = ['userID', 'TrID', 'rowID', 'colID', 'utc', 'lid']
    uids, vectors, graphs = [], [], []
    n, tsum1, tsum2 = 0, 0, 0
    tsize = rlen*clen
    for _, group in trajs:
        n += 1
        uid = group['userID'].iloc[0]
        vector = torch.Tensor(group.drop(columns=drop_columns).values)
        coords = group[['rowID', 'colID']].values.T
        coords[0] = (coords[0]-rmin)//multipler
        coords[1] = (coords[1]-cmin)//multipler
        graph = np.zeros((rlen, clen))
        np.add.at(graph, tuple(coords), 1)
        tsum1 += np.sum(graph)/tsize
        tsum2 += np.sum(graph*graph)/tsize
        graph.resize((1, rlen, clen), refcheck=False)
        uids.append(uid)
        vectors.append(vector)
        graphs.append(torch.Tensor(graph))
    gmean = tsum1/n
    gstd = sqrt(tsum2/n-gmean*gmean)
    for i in range(len(graphs)):
        graphs[i] = (graphs[i]-gmean)/gstd
    return uids, vectors, graphs, len(vectors[0][0])

def tensor_fix_length(x:torch.Tensor, l:int) -> torch.Tensor:
    # x: traj_len * embed_size
    if x.size(0)>l:
        return x[:l]
    pad = torch.zeros((l, x.size(1)))
    pad[:x.shape[0]] = x
    return pad

###################################################
# 已经弃置的代码中使用到的函数
###################################################

def point2grid(dataset:pd.DataFrame, grid_size:float, step:int) -> pd.DataFrame:
    maxlon, minlon = dataset["longitude"].max(), dataset["longitude"].min()
    maxlat, minlat = dataset["latitude"].max(), dataset["latitude"].min()
    midlat = (maxlat + minlat) / 2
    km_per_lon, km_per_lat = 111.32 * math.cos(math.radians(midlat)), 111
    for i in range(1, step+1):
        cellSize = grid_size*i
        deg_lon, deg_lat = cellSize / km_per_lon, cellSize / km_per_lat
        lon_grids, lat_grids = int((maxlon - minlon) / deg_lon) + 1, int((maxlat - minlat) // deg_lat) + 1
        grid_lon, grid_lat = (maxlon - minlon) / lon_grids, (maxlat - minlat) / lat_grids
        dataset[f"colID{i}"] = dataset["longitude"].sub(minlon).floordiv(grid_lon).astype(int)
        dataset[f"rowID{i}"] = dataset["latitude"].sub(minlat).floordiv(grid_lat).astype(int)
    dataset.drop(columns=["longitude", "latitude"], inplace=True)
    dataset["colID"] = dataset["colID1"]
    dataset["rowID"] = dataset["rowID1"]
    return dataset