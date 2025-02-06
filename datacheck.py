import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import raster_tools as rt
import os

# Define the directory path
year = input("Enter the year: ")
base_dir = f"data/wfspreadts_orig/{year}"
# base_dir = 'data/download'

# Dictionary to store fire counts
active_fire_counts = {}

# Iterate through each fire directory
for fire_id in os.listdir(base_dir):
    fire_path = os.path.join(base_dir, fire_id)
    
    if os.path.isdir(fire_path):  # Ensure it's a directory
        no_fire_count = 0
        total_rasters = 0
        
        # Iterate through each raster file in the fire directory
        for file in os.listdir(fire_path):
            if file.endswith(".tif"):
                print(f'checking {fire_id}')
                total_rasters += 1
                raster_path = os.path.join(fire_path, file)
                
                # Load the raster
                raster = rt.Raster(raster_path)
                
                if 'active fire' in raster.xdata.long_name:
                    band_23_mean = raster.get_bands(raster.xdata.long_name.index('active fire')+1).mean().compute()
                
                    # Check if the mean value is NaN or zero (indicating no active fire)
                    if np.isnan(band_23_mean) or band_23_mean == 0:
                        no_fire_count += 1

        # Store the results
        active_fire_counts[fire_id] = {
            "total_rasters": total_rasters,
            "no_fire_rasters": no_fire_count,
            "percentage_no_fire": (no_fire_count / total_rasters) * 100 if total_rasters > 0 else 0
        }

# # Iterate through each fire directory
# for fire in os.listdir(base_dir):
    
#     no_fire_count = 0
#     total_rasters = 0
    
#     # Iterate through each raster file in the fire directory
#     print(f'checking {fire}')
#     if fire.endswith(".tif"):
#         total_rasters += 1
#         raster_path = os.path.join(base_dir, fire)
        
#         # Load the raster
#         raster = rt.Raster(raster_path)
        
#         # Compute the mean of band active fire
#         # check if the raster has the band active fire
#         if 'active fire' in raster.xdata.long_name:
#             band_23_mean = raster.get_bands(raster.xdata.long_name.index('active fire')+1).mean().compute()
        
#             # Check if the mean value is NaN or zero (indicating no active fire)
#             if np.isnan(band_23_mean) or band_23_mean == 0:
#                 no_fire_count += 1
#     # Store the results
#     active_fire_counts[fire] = {
#         "total_rasters": total_rasters,
#         "no_fire_rasters": no_fire_count,
#         "percentage_no_fire": (no_fire_count / total_rasters) * 100 if total_rasters > 0 else 0
#     }

# Display the results
df = pd.DataFrame.from_dict(active_fire_counts, orient="index")
# sum the total number of rasters and the number of rasters with no fire
total_rasters = df["total_rasters"].sum()
no_fire_rasters = df["no_fire_rasters"].sum()

# calculate the percentage of rasters with no fire
percentage_no_fire = (no_fire_rasters / total_rasters) * 100

print(f"Total rasters: {total_rasters}")
print(f"Total rasters with no fire: {no_fire_rasters}")
print(f"Percentage of rasters with no fire: {percentage_no_fire:.2f}%")

# print the fires with the lowest percentage of rasters with no fire
print(df.sort_values(by="percentage_no_fire", ascending=True))