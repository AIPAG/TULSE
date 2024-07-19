import pandas as pd

# 预设的区域范围
presetScope = {
    'brightkite_D': [-76, -73, 39.5, 42],
    'brightkite_S': [-2, 2, 50, 54],
    'foursquare_S': [139.67, 139.82, 35.75, 35.85],
    'foursquare_D': [139.67, 139.82, 35.6, 35.75],
    'geolife_S': [115.25, 117.3, 39.26, 41.03],
    'geolife_D': [116.295, 116.345, 39.965, 40.015],
    'gowalla_S': [-85, -80, 34.4, 37],
    'gowalla_D': [-77.8, -73, 38.6, 41.3]
}

# v-1.1: 将串行修改userID改为通过replace函数并行修改
def rename_uid(DF:pd.DataFrame):
    """从0开始重新编号userID

    Args:
        DF (pd.DataFrame): 轨迹数据集
    """
    userID_list = DF['userID'].unique().tolist()
    IDmap = {uid:i for i, uid in enumerate(userID_list)}
    DF.replace({'userID':IDmap}, inplace=True)

# v-1.1: 筛选轨迹原来采用的是统计需要移除的点然后从数据集中移除，改为直接从数据集中筛选范围内的点
def narrowScope(Allcheckins:pd.DataFrame, minlon=-180.0, maxlon=180.0, minlat=-85.0, maxlat=85.0, threshold_checkins=100) -> pd.DataFrame:
    """根据给定的经纬度范围筛选轨迹点，并筛选位置点数量高于阈值的用户

    Args:
        Allcheckins (pd.DataFrame): 轨迹数据集
        minlon (float, optional): 经度下界. Defaults to -180.0.\n
        maxlon (float, optional): 经度上界. Defaults to 180.0.\n
        minlat (float, optional): 纬度下界. Defaults to -85.0.\n
        maxlat (float, optional): 纬度上界. Defaults to 85.0.\n
        threshold_checkins (int, optional): 用户位置点数量阈值. Defaults to 100.

    Returns:
        pd.DataFrame: 筛选后的轨迹数据集
    """
    Allcheckins = Allcheckins[(Allcheckins['longitude']>=minlon) & (Allcheckins['longitude']<=maxlon) & 
                              (Allcheckins['latitude' ]>=minlat) & (Allcheckins['latitude' ]<=maxlat)]
    g = Allcheckins.groupby(['userID'], as_index=False)
    count_table = g.size().rename(columns={'size': 'num_checkins'})
    retain_user = count_table[count_table['num_checkins'] >= threshold_checkins]
    retain_list = retain_user['userID'].tolist()
    Allcheckins = Allcheckins[Allcheckins['userID'].isin(retain_list)]
    return Allcheckins


def loadDataset(args):
    datasetName = args.dataset
    # ---------------read dataset--------------#
    dataPath = './data/' + datasetName + '.csv'
    allCheckin = pd.read_csv(dataPath, sep=',')

    # -------------narrow the scope----------#
    if args.isDense:
        datasetName = args.dataset + '_D'
    else:
        datasetName = args.dataset + '_S'
    minlon, maxlon, minlat, maxlat = presetScope[datasetName]
    allCheckin = narrowScope(allCheckin, minlon, maxlon, minlat, maxlat, args.threshold)

    # -------------statistic--------------#
    checkinNum = len(allCheckin)
    userNum = len(allCheckin['userID'].drop_duplicates().values.tolist())

    rename_uid(allCheckin)
    allCheckin.reset_index(inplace=True, drop=True)

    return allCheckin, checkinNum, userNum
