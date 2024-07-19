import pandas as pd
import math
import numpy as np
import os
import pyperclip
import torch
import torch.nn as nn

from src.tools.encoder import timeslicing
from src.tools.showinfo import mytimer

class Se(nn.Module):
    def __init__(self, in_channel:int, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel//reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction, out_features=in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x:torch.Tensor):
        out:torch.Tensor = self.pool(x)
        out = self.fc(out.view(out.size(0), -1))
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x
    
train_data = pd.read_csv('./dataset/foursquare_D_train.csv')
minlon, maxlon, minlat, maxlat = 139.67, 139.82, 35.6, 35.75
cellSize = 0.2
midlat = (maxlat + minlat) / 2
km_per_lon, km_per_lat = 111.32 * math.cos(math.radians(midlat)), 111
deg_lon, deg_lat = cellSize / km_per_lon, cellSize / km_per_lat
lon_grids, lat_grids = int((maxlon - minlon) / deg_lon) + 1, int((maxlat - minlat) // deg_lat) + 1
grid_lon, grid_lat = (maxlon - minlon) / lon_grids, (maxlat - minlat) / lat_grids
get_grid = lambda lon, lat: (int((lon-minlon)/grid_lon), int((lat-minlat)/grid_lat))
get_lid = lambda lon, lat: 'X'.join(map(str, get_grid(lon, lat)))

tss = timeslicing(train_data)
get_tss = lambda lon, lat: tss[tss['lid']==get_lid(lon, lat)].to_numpy()
# print(get_lid(139.766, 35.681))
# print(get_tss(139.766, 35.681))
# print('\t'.join([str(i) for i in get_tss(139.766, 35.681)[0]]))
info = '\t'.join([str(i) for i in get_tss(139.7387, 35.7339)[0]])
# info = '\t'.join([str(i) for i in get_tss(139.7387, 35.7339)[0]])
print(info)
pyperclip.copy(info)
