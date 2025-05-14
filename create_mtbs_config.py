import geopandas as gpd
import pandas as pd
from datetime import datetime, timedelta
import yaml

def create_fire_config_mtbs(geojson_path, output_path, year):
    # Read GeoJSON
    gdf = gpd.read_file(geojson_path)
    
    # Convert dates
    gdf['Ig_Date'] = pd.to_datetime(gdf['Ig_Date'])
    gdf['End_Date'] = pd.to_datetime(gdf['End_Date'])
    
    # Filter for year
    gdf = gdf[gdf['YEAR'] == year]
    
    config = {
        'output_bucket': 'firespreadprediction',
        'rectangular_size': 0.5,
        'year': year
    }
    
    class DateSafeYAMLDumper(yaml.SafeDumper):
        def represent_data(self, data):
            if isinstance(data, datetime):
                return self.represent_scalar('tag:yaml.org,2002:timestamp', data.strftime('%Y-%m-%d'))
            return super().represent_data(data)
    
    for idx, row in gdf.iterrows():
        start_date = row['Ig_Date'] - timedelta(days=4)
        end_date = row['End_Date'] + timedelta(days=4)
        
        config[f'fire_{row.Event_ID}'] = {
            'latitude': float(row['BurnBndLat']),
            'longitude': float(row['BurnBndLon']),
            'start': start_date.date(),
            'end': end_date.date()
        }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, Dumper=DateSafeYAMLDumper, default_flow_style=False, sort_keys=False)

# Usage:
YEAR = 2023
create_fire_config_mtbs(f'data/mtbs_perims/mtbs_perimeter_data/mtbs_perims_{YEAR}.geojson', f'WildfireSpreadTSCreateDataset/config/us_fire_{YEAR}_mtbs.yml', YEAR)