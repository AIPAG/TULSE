#查看经纬度范围，选定为某个州
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def rename_uid(DF:pd.DataFrame):
    ''' 从0开始重新编号userID。 '''
    userID_list = DF['userID'].unique().tolist()
    IDmap = {uid:i for i, uid in enumerate(userID_list)}
    DF.replace({'userID':IDmap}, inplace=True)
    return DF

def filt_user(dataset:pd.DataFrame, threshold:int) -> pd.DataFrame:
    names = dataset["userID"].value_counts(sort=False)
    names = names[names>=threshold].index
    return dataset[dataset["userID"].isin(names)]


def coordinate2grid(allCheckins:pd.DataFrame, cellSize: float) -> pd.DataFrame:
    ''' 根据坐标的范围与给定的网格大小（单位：km）划分网格 '''
    # cellsize: 每个格子的长度，单位为km。实际长度与该数值有一定偏差，但不超过1%，因此可以忽视
    maxlon, minlon = allCheckins["longitude"].max(), allCheckins["longitude"].min()
    maxlat, minlat = allCheckins["latitude"].max(), allCheckins["latitude"].min()
    midlat = (maxlat + minlat) / 2
    km_per_lon, km_per_lat = 111.32 * math.cos(math.radians(midlat)), 111
    deg_lon, deg_lat = cellSize / km_per_lon, cellSize / km_per_lat
    lon_grids, lat_grids = int((maxlon - minlon) / deg_lon) + 1, int((maxlat - minlat) // deg_lat) + 1
    grid_lon, grid_lat = (maxlon - minlon) / lon_grids, (maxlat - minlat) / lat_grids
    allCheckins["longitude"] = allCheckins["longitude"].sub(minlon).floordiv(grid_lon).astype(int)
    allCheckins["latitude"] = allCheckins["latitude"].sub(minlat).floordiv(grid_lat).astype(int)
    # allCheckins = allCheckins.rename(columns={"longitude": "colID", "latitude": "rowID"})
    return allCheckins

def cut_analyse(path:str, name:str, minlon:float, maxlon:float, minlat:float, maxlat:float) -> None:
    dataset = pd.read_csv(path + name + ".csv")
    dataset = dataset[(dataset['latitude']<=maxlat) & (dataset['latitude']>=minlat) & 
                      (dataset['longitude']<=maxlon) & (dataset['longitude']>=minlon)]
    # dataset = rename_uid(dataset)
    dataset = filt_user(dataset, 300)
    dataset = coordinate2grid(dataset, 0.05)
    users = dataset['userID'].unique()
    usernum = len(users)
    checkinnum = len(dataset)
    plt.title(f"dataset {name} with {usernum} user and {checkinnum} checkins")
    for user in users:
        userdata = dataset[dataset['userID']==user]
        plt.scatter(userdata['longitude'], userdata['latitude'], marker='.', s=2, c=np.random.rand(3,))
    # plt.scatter(dataset['longitude'], dataset['latitude'], marker='.', s=0.01)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.show()

if __name__ == '__main__':
    datapath = "./data/"

    # analysis_scope(output_csv_file,minlon,maxlon,minlat,maxlat,threshold_checkins)
    cut_analyse(datapath, "gowalla", -77.8, -73, 38.6, 41.3)
