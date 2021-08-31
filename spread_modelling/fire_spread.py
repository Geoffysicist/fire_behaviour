"""
Fire Spread Models. 
Unless otherwise indicated all equations numbers refer to:
Cruz et al. 2015.
"""

import pandas as pd
import csv
import datetime as dt
import math as m
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.series import Series
import matplotlib.pyplot as plt
from matplotlib.path import Path
from typing import Dict, KeysView, List

DATE = 'Local Date'
TIME = 'Local Time'
TEMP = 'Temp (C)'
RH = 'RH (%)'
WIND_DIR = 'Wind Dir'
WIND_SPEED = 'Wind Speed (km/h)'
DF = 'Drought Factor'
FFDI = 'FFDI'
GFDI = 'GFDI'
DATETIME = 'DateTime'
FROS = 'FROS km/h' # forward ROS
ROS = 'flank ROS km/h'


def get_weather(fn):
    '''reads BOM point weather data into a panda df.
    
    args:
        fs (str) path to the csv file
        
    returns
        panda df'''

    df = pd.read_csv(fn, header=6)
    df[DATETIME] = df[[DATE,TIME]].agg(" ".join, axis=1)
    df[DATETIME] = pd.to_datetime(df[DATETIME],infer_datetime_format=True)
    df = df.drop([DATE, TIME],axis=1)

    # for ease of reading put DateTime in thefirst column
    datetime_col = df.pop(DATETIME)
    df.insert(0, DATETIME, datetime_col)

    return df

def trim_weather(weather_df, start_date, start_time, duration):
    """trins BOM weather df to the start time plus the duration, inclusive.

    args:
        weather_df (panda df)
        start_date (str) format %Y%m%d
        start_time (str) format %H:%M
        duration (str) hours

    returns:
        panda df
    """
    start_datetime = start_date + " " + start_time
    start_datetime = dt.datetime.strptime(start_datetime,"%Y%m%d %H:%M")
    finish_datetime = start_datetime + dt.timedelta(hours=duration)
    
    return weather_df[(weather_df[DATETIME] >= start_datetime) & (weather_df[DATETIME] <= finish_datetime)]

def spread_direction(weather_df: DataFrame) -> DataFrame:
    """ Converts wind direction to spread direction"""
    return np.where(
        weather_df[WIND_DIR] < 180,
        weather_df[WIND_DIR] + 180,
        weather_df[WIND_DIR] - 180
    )

def slope_correction(ros: Series, slope: int) -> Series:
    """Adjusts ROS for slope according to Eqn 2.1
    """
    ros = ros * m.exp(0.069*slope)
    return ros

def get_FFDI(weather_df: DataFrame, wind_red: int = 3, flank=False) -> Series:
    """Calculates FFDI from Eqn 5.19.

    if flank calculates the ffdi with wind speed = 0
    """
    if flank:
        wind_speed = 0
    else:
        wind_speed = weather_df[WIND_SPEED]

    ffdi = 2.0*np.exp(
        -0.450 + 0.987*np.log(weather_df[DF])
        -0.0345*weather_df[RH]
        +0.0338*weather_df[TEMP]
        +0.0234* wind_speed * 3 / wind_red #Tolhurst wind reduction
        )
    
    # return np.round(ffdi, 1) 
    return ffdi   

#### FIRE SPREAD MODELS #####
def ros_grass_cheney(weather_df: DataFrame, grass_state: str, grass_curing):
    """Cheney et al. 1998
    inputs: 
        Wind speed 10m (km/h)
        Temperature (oC)
        Relative Humidity (%)
        Curing level (%)
        Grass state - natural (N), grazed (G), eaten out (E), (W) Woodlands, (F) Open forest
    """
    grass_state = grass_state.upper()
    # TODO raise error curing is out of range 20 - 100
    if grass_curing <= 20:
        grass_curing = 20
        # raise ValueError(f'Curing coefficient {grass_curing} outside model range 20 < C < 100')

    # dead fuel moisture content from weather data Eqn 3.8
    dead_fuel_mc = 9.58 - 0.205*weather_df[TEMP] + 0.138 *weather_df[RH]
    grass_df = dead_fuel_mc.to_frame(name='dead_fuel_mc')
    
    # fuel moisture coeff Eqn 3.7
    grass_df['fm_coeff'] = np.where(
        grass_df['dead_fuel_mc'] < 12,
        np.exp(-0.108*grass_df['dead_fuel_mc']),
        np.where(
            weather_df[WIND_SPEED] < 10,
            0.684 - 0.0342*grass_df['dead_fuel_mc'],
            0.547 - 0.0228*grass_df['dead_fuel_mc']
        )
    )
    
    # curing coefficient from Eqn 3.10
    curing_coeff = 1.036/(1+103.99*m.exp(-0.0996*(grass_curing-20)))

    # create the ros dataframe from the datetime
    ros_df = weather_df[DATETIME].to_frame(name='DateTime')
    ros_df['Direction'] = spread_direction(weather_df)

    #ros
    if grass_state in 'NWF':
        # Eqn 3.5
        ros_df[FROS] = np.where(
            weather_df[WIND_SPEED] > 5,
            (1.4 + 0.838*(weather_df[WIND_SPEED] - 5)**0.844)*grass_df['fm_coeff']*curing_coeff,
            (0.054 + 0.269*weather_df[WIND_SPEED])*grass_df['fm_coeff']*curing_coeff
        )

        if grass_state == 'W':
             ros_df[FROS] = ros_df[FROS] * 0.5
        elif grass_state == 'O':
            ros_df[FROS] = ros_df[FROS] * 0.3
        
        ros_df[ROS] = np.where(
            weather_df[WIND_SPEED] > 5,
            (0.054 + 0.269*5)*grass_df['fm_coeff']*curing_coeff,
            (0.054 + 0.269*weather_df[WIND_SPEED])*grass_df['fm_coeff']*curing_coeff        
        )

    elif grass_state in 'GE':
        # Eqn 3.6
        ros_df[FROS] = np.where(
            weather_df[WIND_SPEED] > 5,
            (1.1 + 0.715*(weather_df[WIND_SPEED] - 5)**0.844)*grass_df['fm_coeff']*curing_coeff,
            (0.054 + 0.209*weather_df[WIND_SPEED])*grass_df['fm_coeff']*curing_coeff
        )
        ros_df[ROS] = np.where(
            weather_df[WIND_SPEED] > 5,
            (0.054 + 0.209*5)*grass_df['fm_coeff']*curing_coeff,
            (0.054 + 0.209*weather_df[WIND_SPEED])*grass_df['fm_coeff']*curing_coeff
        )
        if grass_state == 'E':
            # Cruz et al. argue that should be half of G but no studies
            ros_df[FROS] /= 2
            ros_df[ROS] /= 2
        
    else:
        raise ValueError('Not a valid grass state')

    ros_df[FROS] = np.round(ros_df[FROS],2)
    ros_df[ROS] = np.round(ros_df[ROS],2)
    return ros_df

def ros_forest_mk5(weather_df: DataFrame, fuel_load: int, wind_red: int) -> DataFrame:
    """McArthur 1973a Mk5 Forest Fire Danger Meter
    """
    ros_df = weather_df[DATETIME].to_frame(name='DateTime')
    ros_df['Direction'] = spread_direction(weather_df)
    ros_df[FFDI] = get_FFDI(weather_df, wind_red)

    ros_df[FROS] = 0.0012*ros_df[FFDI]*fuel_load
    ros_df[ROS] = 0.0012*get_FFDI(weather_df, flank=True)*fuel_load

    return ros_df

def plot_paths(ros_dfs: Dict):
    """Prduces a vector plot of the path of the fire from FROS model.
    """
    axes = plt.axes()
    colors = ['blue', 'red']
    color_id = 1

    for model,ros_df in ros_dfs.items():
        times = list(ros_df[DATETIME])
        direction = list(ros_df['Direction'])
        fros = list(ros_df[FROS])

        x = y = 0
        arrows = []

        for i in range(len(times)-1):
            time_interval = (times[i+1] - times[i]).total_seconds()/3600
            angle = 90 - (direction[i] - 360)
            if angle >= 360: angle -= 360
            angle = m.radians(angle)
            length = int(fros[i] * time_interval * 1000) # convert to metres
            dx = int(length * m.cos(angle))
            dy = int(length * m.sin(angle))
            arrows.append([x, y, dx, dy])
            x += dx
            y += dy

        
        for a in arrows:
            # axes.arrow(*a, color="k", head_width=16, head_length=128, overhang=1, length_includes_head=True)
            axes.arrow(*a,  length_includes_head=True, color=colors[color_id])
            color_id ^= 1

    plt.show()

if __name__ == "__main__":
    # general model settings
    weather_fn = 'data\TestPointForecast.csv'
    start_date = '20210827'
    start_time = '09:00'
    duration = 24 #hours
    slope = 10 #but note Cruz et al. for large fires slope effect negligible

    # Select the models you wan to run by assigning them 'True'
    selected_models = {
        'GRASS: Cheney et al. 1998': True,
        'FOREST: McArthur Mk5': True
    }

    # model specific data
    # grass state # N - natural, G - grazed, E - eaten out
    #   W - woodland (canopy cover < 30%),
    #   F - Open forest (canopy cover 30-70%, 10-15 m tall)
    grass_state = 'E' 
    grass_curing = 85 # per cent should between 20 and 100

    #forest
    fuel_load = 20 # t/ha
    wind_reduction = 3 # Tolhurst's wind reduction factor between 1 - 6


    ###################################
    ###### DO NOT EDIT BELOW HERE #####
    ###################################
    weather_df = get_weather(weather_fn)
    weather_df = trim_weather(weather_df, start_date, start_time, duration)

    MODELS = {
        'GRASS: Cheney et al. 1998': ros_grass_cheney(weather_df, grass_state, grass_curing),
        'FOREST: McArthur Mk5': ros_forest_mk5(weather_df, fuel_load, wind_reduction)
    }

    model_outputs = {} #model name as key, dataframes as val
    
    # grass
    # ros_grass = ros_grass(weather_df, grass_state, grass_curing)
    # print(ros_grass)

    # forest macarthur 77
    # ros_forest_mk5 = ros_forest_mk5(weather_df, fuel_load, wind_reduction)
    # ros_forest_mk5[FROS] = slope_correction(ros_forest_mk5[FROS], slope)
    # print(ros_forest_mk5)

    for key, val in selected_models.items():
        if val:
            model_outputs[key] = MODELS[key]
    

    for key, val in model_outputs.items():
        print(key)
        print(val)
        print('\n')

    plot_paths(model_outputs)

    print('fire spread done')
