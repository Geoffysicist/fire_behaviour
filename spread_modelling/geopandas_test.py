import geopandas as gpd
# import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from shapely.geometry import Point
from fiona.crs import from_epsg

# Set filepath
fp = "data\\test_path.shp"
# # Read file using gpd.read_file()
# geodata = gp.read_file(fp)

ignition_df = DataFrame({'lat':[-30.80132], 'lon':[152.97958], 'elev': [30]})
ignition_df = ignition_df.append(DataFrame({'lat':[-30.810001], 'lon':[153.00682], 'elev': [9]}))

ignition_gdf = gpd.GeoDataFrame(
    ignition_df, 
    geometry=gpd.points_from_xy(ignition_df.lon, ignition_df.lat),
    crs="EPSG:4283"
)

print(ignition_gdf.crs)

#convert to MGA Zone 55
# ignition_gdf_mga = ignition_gdf.to_crs("EPSG:28355")
#convert to MGA Zone 56
ignition_gdf_mga = ignition_gdf.to_crs("EPSG:28356")

fig, ax = plt.subplots(1, 1)


ignition_gdf_mga.plot(ax=ax, legend=True)

plt.show()
print(ignition_gdf)
print(ignition_gdf_mga)
# print(newdata.crs)