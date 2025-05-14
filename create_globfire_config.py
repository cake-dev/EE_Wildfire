import geopandas as gpd
import pandas as pd
from datetime import datetime, timedelta
import yaml

def create_fire_config_globfire(geojson_path, output_path, year):
    gdf = gpd.read_file(geojson_path)
    gdf['IDate'] = pd.to_datetime(gdf['IDate'], unit='ms')
    gdf['FDate'] = pd.to_datetime(gdf['FDate'], format='mixed')
    
    gdf = gdf[gdf['IDate'].dt.year == year]
    first_occurrences = gdf.sort_values('IDate').groupby('Id').first()
    last_occurrences = gdf.sort_values('IDate').groupby('Id').last()
    
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
    
    for idx in first_occurrences.index:
        first = first_occurrences.loc[idx]
        last = last_occurrences.loc[idx]
        
        end_date = last['FDate'] if pd.notna(last['FDate']) else last['IDate']
        start_date = first['IDate'] - timedelta(days=4)
        end_date = end_date + timedelta(days=4)
        
        config[f'fire_{idx}'] = {
            'latitude': float(first['lat']),
            'longitude': float(first['lon']),
            'start': start_date.date(),
            'end': end_date.date()
        }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, Dumper=DateSafeYAMLDumper, default_flow_style=False, sort_keys=False)

# Usage:
YEAR = 2020
create_fire_config_globfire(f'data/perims/combined_fires_{YEAR}.geojson', f'config/us_fire_{YEAR}_1e7_test.yml', YEAR)