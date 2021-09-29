"""
Fire Spread Models. 
Unless otherwise indicated all equations numbers refer to:
Cruz et al. 2015.

agnostic to slope ATM. Use discretion when plotting.
note Cruz et al. for large fires slope effect negligible
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
MC = 'Moisture content %'
MF = 'Moisture function'
DATETIME = 'DateTime'
FROS = 'FROS (km/h)' # forward ROS
ROS = 'flank ROS (km/h)'
IG_TIME = 'Ignition time'
IG_COORDS = 'Ignition coordinates'
FROS_DIR = 'Direction (\u00b0)'
VEC = 'FROS vector (m)'
PATHS = 'Path coords'

COORD_SYSTEMS = {
    'GDA94_LL': 'EPSG:4283',
    'WGS84_LL': 'EPSG:4326',
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

def get_weather(fn: str, header_row: int) -> DataFrame:
    '''reads BOM point weather data into a panda df.
    
    args:
        fs (str) path to the csv file
        
    returns
        panda df'''

    header_row -= 1 # 0 indexed
    df = pd.read_csv(fn, header=header_row)
    df[DATETIME] = df[[DATE,TIME]].agg(" ".join, axis=1)
    df[DATETIME] = pd.to_datetime(df[DATETIME],infer_datetime_format=True)
    df = df.drop([DATE, TIME],axis=1)

    # for ease of reading put DateTime in thefirst column
    datetime_col = df.pop(DATETIME)
    df.insert(0, DATETIME, datetime_col)

    # remove spurious precision of wind direction
    df[WIND_DIR] = df[WIND_DIR].astype(int)

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

def weather_to_amicus_csv(weather_df: DataFrame, output_fn) -> None:
    """Exports the weather dataframe as a csv file formatted for importing into Amicus."""
    amicus_df = weather_df[DATETIME].to_frame(name='Date time')
    amicus_df['Air temperature (\u00b0C)'] = weather_df[TEMP]
    amicus_df['Relative humidity (%)'] = weather_df[RH]
    amicus_df['10 m wind speed (km/h)'] = weather_df[WIND_SPEED]
    amicus_df['Wind direction (\u00b0)'] = weather_df[WIND_DIR]
    amicus_df['Cloud cover (%)'] = 0
    output_fn = f'{output_fn}_amicus_weather.csv'
    amicus_df.to_csv(output_fn, index=False, date_format="%d/%m/%Y %H:%M", encoding='ANSI')


def spread_direction(weather_df: DataFrame) -> DataFrame:
    """ Converts wind direction to spread direction"""

    return np.where(
        weather_df[WIND_DIR] < 180,
        weather_df[WIND_DIR] + 180,
        weather_df[WIND_DIR] - 180
    )

def slope_correction(ros_df: DataFrame, slope: int) -> DataFrame:
    """Adjusts ROS for slope according to Eqn 2.1

    not used ATM
    """
    ros_df[FROS] = ros_df[FROS] * m.exp(0.069*slope)
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


#### post processing and output  ####
def post_process(ros_df):
    """Rounds values to 2 decimal places and adds ROS vectors.
    """
    # ros_df[FROS] = np.round(ros_df[FROS],2)
    # ros_df[ROS] = np.round(ros_df[ROS],2)

    # get rid of the spurious precision
    precision_dict = {
        0: [FFDI],
        1: [MC, MF],
        2: [FROS, ROS]
    }

    for precision, fields in precision_dict.items():
        for field in fields:
            try:
                ros_df[field] = np.round(ros_df[field],precision)
                if not precision:
                    ros_df[field] = ros_df[field].astype(int)
            except KeyError:
                pass # I know this is bad but do you have a better suggestion?
            except Exception as e:
                print(f'post_process error {type(e)}: {e}')

    # for key in [MC, MF]:
    #     try:
    #         ros_df[key] = np.round(ros_df[key],1)
    #     except KeyError:
    #         pass # I know this is bad but do you have a better suggestion?
    #     except Exception as e:
    #         print(f'post_process error {type(e)}: {e}')

    # for key in [FROS, ROS]:
    #     try:
    #         ros_df[key] = np.round(ros_df[key],2)
    #     except KeyError:
    #         pass # I know this is badbut do you have a better suggestion?
    #     except Exception as e:
    #         print(f'post_process error {type(e)}: {e}')


    #calculate the magnitude of the fros vectors
    # TODO move this to paths...maybe
    times = list(ros_df[DATETIME])
    direction = list(ros_df[FROS_DIR])
    fros = list(ros_df[FROS])
    fros_vectors = [0] # first vector is 0 ie starting point
    for i, time in enumerate(times[:-1]):
        time_interval = (times[i+1] - time).total_seconds()/3600 # convert time interval to hours
        # TODO store fros vector as complex number and add vectors 
        fros_vectors.append(int(fros[i] * time_interval * 1000)) # convert to metres
    
    # fros_vectors.append(0)
    ros_df[VEC] = fros_vectors

    return ros_df

def save_csvs(ros_df_dict: Dict, output_fn):
    for model, df in ros_df_dict.items():
        df.to_csv(f'{output_fn}_{model}.csv', index=False, encoding='ANSI')

def create_path_gdf(ros_df: DataFrame, ignition_date: str, ignition_time: str, ignition_coords: List):
    """Create a GeoDataFrame of the spread path for plotting and shapefile creation."""

    # convert to grid projection for path calculations
    ignition_gdf = coords_to_gdf(ignition_coords, ignition_date, ignition_time)
    ignition_gdf = reproject(ignition_gdf,'MGA94_56')

    vec_dir = list(ros_df[FROS_DIR])
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
    #TODO move this to an export function
    # shapefiles dont support datetimes

    for model, gdf in gdf_dict.items():
        gdf[DATETIME] = np.datetime_as_string(gdf[DATETIME], unit='m') # minute precision
        gdf.to_file(f'{output_fn}_{model}.shp')

def plot_paths(gdf_dict: Dict) -> None:
    """Prduces a vector plot of the path of the fire from FROS model.
    """
    fig, ax = plt.subplots(1, 1)

    # TODO change so that get all models plotted
    for model, gdf in gdf_dict.items():
        gdf.plot(ax=ax, legend=True)

    plt.show()
    return None


#### FIRE SPREAD MODELS #####
def ros_grass_cheney(weather_df: DataFrame, params):
# def ros_grass_cheney(weather_df: DataFrame, grass_state: str, grass_curing: int):
    """Cheney et al. 1998
    inputs: 
        from weather dataframe:
          Wind speed 10m (km/h)
          Temperature (oC)
          Relative Humidity (%)
        from params dict:
            Curing level (%)
            Grass state - natural (N), grazed (G), eaten out (E), (W) Woodlands, (F) Open forest
    """
    grass_state = params['grass_state'].upper()
    grass_curing = params['grass_curing']
    if grass_curing <= 20:
        grass_curing = 20
    
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
    ros_df[FROS_DIR] = spread_direction(weather_df)

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

    return post_process(ros_df)

def ros_forest_mk5(weather_df: DataFrame, params: Dict) -> DataFrame:
    """McArthur 1973a Mk5 Forest Fire Danger Meter
    """
    ros_df = weather_df[DATETIME].to_frame(name='DateTime')
    ros_df[FROS_DIR] = spread_direction(weather_df)
    ros_df[FFDI] = get_FFDI(weather_df, params['wind_reduction'])

    ros_df[FROS] = 0.0012*ros_df[FFDI]*params['fuel_load']
    ros_df[ROS] = 0.0012*get_FFDI(weather_df, flank=True)*params['fuel_load']

    return post_process(ros_df)

def ros_forest_vesta(weather_df: DataFrame, params: Dict) -> DataFrame:
    """Project Vesta Cheney et al 2012.

    using fuel hazard scores Eq 5.28
    """

    ros_df = weather_df[DATETIME].to_frame(name='DateTime')
    ros_df[FROS_DIR] = spread_direction(weather_df)

    # determine moisture content
    ros_df[MC] = np.where(
        (weather_df[DATETIME].dt.hour >= 9) & (weather_df[DATETIME].dt.hour < 20),
        np.where(
            (weather_df[DATETIME].dt.hour >= 12) & (weather_df[DATETIME].dt.hour < 17), 
            2.76 + (0.124*weather_df[RH]) - (0.0187*weather_df[TEMP]), 
            3.6 + (0.169*weather_df[RH]) - (0.045*weather_df[TEMP])
        ),
        3.08 + (0.198*weather_df[RH]) - (0.0483*weather_df[TEMP])
    )

    # determine moisture function
    ros_df[MF] = 18.35 * ros_df[MC]**-1.495

    # determine the ROS
    ros_df[ROS] = 30.0 * ros_df[MF] / 1000
    ros_df[FROS] = np.where(
        weather_df[WIND_SPEED] > 5,
        30.0 + 1.531 * (weather_df[WIND_SPEED]-5)**0.8576 * params['fhs_surface']**0.93 * (params['fhs_near_surface']*params['fuel_height_ns_cm'])**0.637 * 1.03,
        30
    )

    ros_df[FROS] = ros_df[FROS]* ros_df[MF] / 1000
    return post_process(ros_df)

def ros_forest_vesta_fhr(weather_df: DataFrame, params: Dict) -> DataFrame:
    """Project Vesta Cheney et al 2012.

    using fuel hazard scores Eq 5.31
    """
    # fuel hazard ratings need tobe converted to coefficients
    NEAR_SURFACE = {'L': 0.4694, 'M': 0.7070, 'H': 1.2772, 'V': 1.7492, 'E': 1.2446}
    SURFACE = {'L': 0.0, 'M': 1.5608, 'H': 2.1412, 'V': 2.0548, 'E': 2.3251}
    
    ros_df = weather_df[DATETIME].to_frame(name='DateTime')
    ros_df[FROS_DIR] = spread_direction(weather_df)

    surface_coeff = SURFACE[params['fhr_surface']]
    near_surf_coeff = NEAR_SURFACE[params['fhr_near_surface']]

    # determine moisture content
    ros_df[MC] = np.where(
        (weather_df[DATETIME].dt.hour >= 9) & (weather_df[DATETIME].dt.hour < 20),
        np.where(
            (weather_df[DATETIME].dt.hour >= 12) & (weather_df[DATETIME].dt.hour < 17), 
            2.76 + (0.124*weather_df[RH]) - (0.0187*weather_df[TEMP]), 
            3.6 + (0.169*weather_df[RH]) - (0.045*weather_df[TEMP])
        ),
        3.08 + (0.198*weather_df[RH]) - (0.0483*weather_df[TEMP])
    )

    # determine moisture function
    ros_df[MF] = 18.35 * ros_df[MC]**-1.495

    # determine the ROS
    ros_df[ROS] = 30.0 * ros_df[MF] / 1000
    ros_df[FROS] = np.where(
        weather_df[WIND_SPEED] > 5,
        30.0 + 2.3117 * (weather_df[WIND_SPEED]-5)**0.8364 * m.exp(surface_coeff+near_surf_coeff) * 1.02,
        30
    )
    ros_df[FROS] = ros_df[FROS]* ros_df[MF] / 1000

    return post_process(ros_df)

def ros_forest_vesta_kt(weather_df: DataFrame, params: Dict) -> DataFrame:
    """Project Vesta Cheney et al 2012.

    using my best interpretation of KT spreadsheet
    """
    # fuel hazard ratings need tobe converted to coefficients
    # rate_to_score = {'L': 1, 'M': 2, 'H': 3, 'V': 3.5, 'E': 4}
    
    ros_df = weather_df[DATETIME].to_frame(name='DateTime')
    ros_df[FROS_DIR] = spread_direction(weather_df)

    # determine moisture content
    ros_df[MC] = np.where(
        (weather_df[DATETIME].dt.hour >= 9) & (weather_df[DATETIME].dt.hour < 20),
        np.where(
            (weather_df[DATETIME].dt.hour >= 12) & (weather_df[DATETIME].dt.hour < 17), 
            2.76 + (0.124*weather_df[RH]) - (0.0187*weather_df[TEMP]), 
            3.6 + (0.169*weather_df[RH]) - (0.045*weather_df[TEMP])
        ),
        3.08 + (0.198*weather_df[RH]) - (0.0483*weather_df[TEMP])
    )

    # determine moisture function
    ros_df[MF] = 18.35 * ros_df[MC]**-1.495

    # determine the ROS
    ros_df[ROS] = 30.0 * ros_df[MF] / 1000
    ros_df[FROS] = np.where(
        weather_df[WIND_SPEED] > 5,
        30.0 + 3.102 * (weather_df[WIND_SPEED]-5)**0.904 * m.exp(0.279*params['fhs_surface']+0.611*params['fhs_near_surface']+0.013*params['fuel_height_ns_cm']),
        30
    )

    ros_df[FROS] = ros_df[FROS]* ros_df[MF] / 1000
    return post_process(ros_df)


#### MODEL EXECUTION ####
def run_models(
    weather_fn: str,
    weather_header_row: int,
    start_date: str,
    start_time: str,
    duration: int,
    selected_models: Dict,
    params_grass: Dict,
    params_mk5: Dict,
    params_vesta: Dict
    ) -> Dict:

    """this is where sh*t gets real."""
    start = dt.datetime.now()
    weather_df = get_weather(weather_fn, weather_header_row)
    weather_df = trim_weather(weather_df, start_date, start_time, duration)

    MODELS = {
        # 'GRASS_Cheney_98': ros_grass_cheney(weather_df, grass_state, grass_curing),
        'GRASS_Cheney_98': ros_grass_cheney(weather_df, params_grass),
        'FOREST_Mk5': ros_forest_mk5(weather_df, params_mk5),
        'FOREST_Vesta': ros_forest_vesta(weather_df, params_vesta),
        'FOREST_Vesta_FHR': ros_forest_vesta_fhr(weather_df, params_vesta_fhr),
        'FOREST_Vesta_KT': ros_forest_vesta_kt(weather_df, params_vesta),
    }

    model_outputs = {} # model name as key, dataframes as val

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
    weather_fn = 'input\\test1_PointForecast.csv'
    weather_header_row = 7
    start_date = '20000108'
    start_time = '16:00'
    ignition_date = start_date
    ignition_time = start_time
    ignition_coords = [-34.8350, 148.4186, 'GDA94_LL'] #GDA94_LL or MGA94_Zxx where xx = zone
    duration = 17 #hours
    path_output_fn = 'output\\test1'

    # Select the models you want to run by assigning them 'True'
    selected_models = {
        'GRASS_Cheney_98': True,
        'FOREST_Mk5': True,
        'FOREST_Vesta': True,
        'FOREST_Vesta_FHR': True,
        'FOREST_Vesta_KT': True
    }

    # model specific data
    # grass state # N - natural, G - grazed, E - eaten out
    #   W - woodland (canopy cover < 30%),
    #   F - Open forest (canopy cover 30-70%, 10-15 m tall)
    # curing per cent should between 20 and 100
    params_grass = {
        'grass_state': 'G',
        'grass_curing': 85
    }
    # grass_state = 'W' 
    # grass_curing = 95 # per cent should between 20 and 100

    #forest MK5
    params_mk5 = {
        'fuel_load': 5, #t/ha
        'wind_reduction': 3 # Tolhurst's wind reduction factor between 1 - 6
    }

    #forst Vesta
    # forest vesta
    params_vesta = {
        'fhs_surface': 3.5,
        'fhs_near_surface': 3,
        'fhs_elevated': 2,
        'fuel_height_ns_cm': 25,
        'fuel_height_e_m': 1.5
    }

    params_vesta_fhr = {
        'fhr_surface': 'V',
        'fhr_near_surface': 'H',
    }

    ###################################
    ###### DO NOT EDIT BELOW HERE #####
    ###################################
    weather_df = get_weather(weather_fn, weather_header_row)
    weather_df = trim_weather(weather_df, start_date, start_time, duration)
    weather_to_amicus_csv(weather_df, path_output_fn)


    model_outputs = run_models(
        weather_fn,
        weather_header_row,
        start_date,
        start_time,
        duration,
        selected_models,
        params_grass,
        params_mk5,
        params_vesta
    )

    # Print tables of the models
    for key, val in model_outputs.items():
        print(key)
        print(val)
        print('\n')

    save_csvs(model_outputs,path_output_fn)

    # # do this after printing the models so dont get linestrings
    # model_gdfs = get_gdfs(model_outputs, ignition_date, ignition_time, ignition_coords)

    # # Save shapefile of the fire path
    # # save_shapefiles(model_gdfs, path_output_fn)

    # # Show simple plot of the model
    # # plot_paths(model_outputs, ignition_date, ignition_time, ignition_coords)
    # plot_paths(model_gdfs)

    print('fire spread done')
