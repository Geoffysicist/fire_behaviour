"""
Fire Spread Models. 
Unless otherwise indicated all equations numbers refer to:
Cruz et al. 2015.
"""

# from geopandas import geodataframe
import pandas as pd
import csv
import datetime as dt
import math as m
import numpy as np
from pandas.core.base import NoNewAttributesMixin
from pandas.core.frame import DataFrame
from pandas.core.series import Series
import matplotlib.pyplot as plt
# from matplotlib.path import Path
from typing import Dict, KeysView, List
from geopandas import GeoDataFrame, points_from_xy
from shapely.geometry import Point, LineString


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
FROS = 'FROS (km/h)' # forward ROS
ROS = 'flank ROS (km/h)'
IG_TIME = 'Ignition time'
IG_COORDS = 'Ignition coordinates'
DIR = 'Direction (o)'
VEC = 'FROS Vector (m)'
PATHS = 'Path coords'

COORD_SYSTEMS = {
    'GDA94_LL': 'EPSG:4283',
    'MGA94_49': 'EPSG:28349',
    'MGA94_50': 'EPSG:28350',
    'MGA94_51': 'EPSG:28351',
    'MGA94_52': 'EPSG:28352',
    'MGA94_53': 'EPSG:28353',
    'MGA94_54': 'EPSG:28354',
    'MGA94_55': 'EPSG:28355',
    'MGA94_56': 'EPSG:28356',
}


def reproject(gdf: GeoDataFrame, projection: str):
    'Change gdf to new projection'

    return gdf.to_crs(COORD_SYSTEMS[projection])

def coords_to_gdf(coords: List, date: str, time: str) -> GeoDataFrame:
    start_datetime = date + " " + time
    start_datetime = dt.datetime.strptime(start_datetime,"%Y%m%d %H:%M")
    df = DataFrame({'Ignition time': [start_datetime], 'lon':[coords[1]], 'lat':[coords[0]]})
    gdf = GeoDataFrame(
        df, 
        geometry=points_from_xy(df.lon, df.lat),
        crs=COORD_SYSTEMS[coords[2]]
    )
   
    return gdf

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

def slope_correction(ros_df: DataFrame, slope: int) -> DataFrame:
    """Adjusts ROS for slope according to Eqn 2.1
    """
    ros_df[FROS] = ros_df[FROS] * m.exp(0.069*slope)
    return ros_df

def post_process(ros_df, slope):
    ros_df = slope_correction(ros_df, slope)
    ros_df[FROS] = np.round(ros_df[FROS],2)
    ros_df[ROS] = np.round(ros_df[ROS],2)

    #calculate the magnitude of the fros vectors
    times = list(ros_df[DATETIME])
    direction = list(ros_df[DIR])
    fros = list(ros_df[FROS])
    fros_vectors = []
    for i in range(len(times)-1):
        time_interval = (times[i+1] - times[i]).total_seconds()/3600
        fros_vectors.append(int(fros[i] * time_interval * 1000)) # convert to metres
    
    fros_vectors.append(0)
    ros_df[VEC] = fros_vectors

    return ros_df

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
def ros_grass_cheney(weather_df: DataFrame, grass_state: str, grass_curing: int, slope: int):
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
    ros_df[DIR] = spread_direction(weather_df)

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

    return post_process(ros_df, slope)

def ros_forest_mk5(weather_df: DataFrame, fuel_load: int, wind_red: int, slope: int) -> DataFrame:
    """McArthur 1973a Mk5 Forest Fire Danger Meter
    """
    ros_df = weather_df[DATETIME].to_frame(name='DateTime')
    ros_df[DIR] = spread_direction(weather_df)
    ros_df[FFDI] = get_FFDI(weather_df, wind_red)

    ros_df[FROS] = 0.0012*ros_df[FFDI]*fuel_load
    ros_df[ROS] = 0.0012*get_FFDI(weather_df, flank=True)*fuel_load

    return post_process(ros_df, slope)

# post processing and output
def create_path_gdf(ros_df: DataFrame, ignition_date: str, ignition_time: str, ignition_coords: List):
    """Create a GeoDataFrame of the spread path for plotting and shapefile creation."""

    # convert to grid projection for path calculations
    ignition_gdf = coords_to_gdf(ignition_coords, ignition_date, ignition_time)
    ignition_gdf = reproject(ignition_gdf,'MGA94_56')

    vec_dir = list(ros_df[DIR])
    vec_mag = list(ros_df[VEC])

    x = list(ignition_gdf['geometry'].x)[0]
    y = list(ignition_gdf['geometry'].y)[0]
    geometry = []

    for i in range(len(vec_dir)):

        # geo angles are measured CW from N, trig angles are measures ACW fro x axis
        angle = 90 - (vec_dir[i] - 360)
        if angle >= 360: angle -= 360
        angle = m.radians(angle)

        dx = int(vec_mag[i] * m.cos(angle))
        dy = int(vec_mag[i] * m.sin(angle))
        geometry.append(LineString([Point(x, y), Point(x+dx, y+dy)]))
        x += dx
        y += dy
    
    ros_df['geometry'] = geometry
    ros_gdf = GeoDataFrame(ros_df, geometry='geometry', crs=COORD_SYSTEMS['MGA94_56'])
    ros_gdf = reproject(ros_gdf, ignition_coords[2]) #change back to original projection

    return ros_gdf

def get_gdfs(ros_dfs: Dict, ignition_date: str, ignition_time: str, ignition_coords: List) -> Dict:
    """Prduces GeoDataFrames with the path of the fire from FROS model.
    """
    gdf_dict = {}
    for model, ros_df in ros_dfs.items():
        
        gdf_dict[model] = create_path_gdf(ros_df, ignition_date, ignition_time, ignition_coords)

    return gdf_dict

def save_shapefiles(gdf_dict, output_fn):
    #TODO move tihs to an export function
    # shapefiles dont support datetimes

    for model, gdf in gdf_dict.items():
        gdf[DATETIME] = np.datetime_as_string(gdf[DATETIME], unit='m') # minute precision
        gdf.to_file(f'{output_fn}_{model}.shp')

def plot_paths(gdf_dict: Dict) -> None:
    """Prduces a vector plot of the path of the fire from FROS model.
    """
    fig, ax = plt.subplots(1, 1)

    for model, gdf in gdf_dict.items():
        gdf.plot(ax=ax, legend=True)

    plt.show()
    return None

def run_models(
    weather_fn: str,
    start_date: str,
    start_time: str,
    duration: int,
    slope: int,
    selected_models: Dict,
    grass_state: str,
    grass_curing: int,
    fuel_load: int,
    wind_reduction: int) -> Dict:

    """this is where sh*t gets real."""
    start = dt.datetime.now()
    weather_df = get_weather(weather_fn)
    weather_df = trim_weather(weather_df, start_date, start_time, duration)

    MODELS = {
        'GRASS_Cheney_98': ros_grass_cheney(weather_df, grass_state, grass_curing, slope),
        'FOREST_Mk5': ros_forest_mk5(weather_df, fuel_load, wind_reduction, slope)
    }

    model_outputs = {} #model name as key, dataframes as val

    models_run = 0
    for key, val in selected_models.items():
        if val:
            model_outputs[key] = MODELS[key]
            models_run += 1

    time_elapsed = dt.datetime.now()-start
    print(f'{models_run} models run in {time_elapsed}')
    return model_outputs


if __name__ == "__main__":
    # TODO change model settings to dictionaries
    # general model settings
    weather_fn = 'data\\2000-01-08-XX-XX-XX_PointForecast.csv'
    start_date = '20000108'
    start_time = '16:00'
    ignition_date = start_date
    ignition_time = start_time
    ignition_coords = [-34.8350, 148.4186, 'GDA94_LL'] #GDA94_LL or MGA94_Zxx where xx = zone
    duration = 17 #hours
    slope = 0 #but note Cruz et al. for large fires slope effect negligible
    path_output_fn = 'test1'

    # Select the models you want to run by assigning them 'True'
    selected_models = {
        'GRASS_Cheney_98': True,
        'FOREST_Mk5': True
    }

    # model specific data
    # grass state # N - natural, G - grazed, E - eaten out
    #   W - woodland (canopy cover < 30%),
    #   F - Open forest (canopy cover 30-70%, 10-15 m tall)
    grass_state = 'W' 
    grass_curing = 95 # per cent should between 20 and 100

    #forest
    fuel_load = 5 # t/ha
    wind_reduction = 3 # Tolhurst's wind reduction factor between 1 - 6


    ###################################
    ###### DO NOT EDIT BELOW HERE #####
    ###################################
    model_outputs = run_models(
        weather_fn,
        start_date,
        start_time,
        duration,
        slope,
        selected_models,
        grass_state,
        grass_curing,
        fuel_load,wind_reduction
    )

    # Print tables of the models
    for key, val in model_outputs.items():
        print(key)
        print(val)
        print('\n')

    # do this after printing the models so dont get linestrings
    model_gdfs = get_gdfs(model_outputs, ignition_date, ignition_time, ignition_coords)

    # Save shapefile of the fire path
    save_shapefiles(model_gdfs, path_output_fn)

    # Show simple plot of the model
    # plot_paths(model_outputs, ignition_date, ignition_time, ignition_coords)
    plot_paths(model_gdfs)

    print('fire spread done')
