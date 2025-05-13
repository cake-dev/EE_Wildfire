import ee

# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize(project='ee-earthdata')

import ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import datetime

# Initialize the Earth Engine API
ee.Initialize()

# Define the geometry for contiguous USA
usa_coords = [
    [-125.1803892906456, 35.26328285844432],
    [-117.08916345892665, 33.2311514593429],
    [-114.35640058749676, 32.92199940444295],
    [-110.88773544819885, 31.612036247094473],
    [-108.91086200144109, 31.7082477979397],
    [-106.80030780089378, 32.42079476218232],
    [-103.63413436750255, 29.786401496314422],
    [-101.87558377066483, 30.622527701868453],
    [-99.40039768482492, 28.04018292597704],
    [-98.69085295525215, 26.724810345780593],
    [-96.42355704777482, 26.216515704595633],
    [-80.68508661702214, 24.546812350183075],
    [-75.56173032587596, 26.814533788629998],
    [-67.1540159827795, 44.40095539443753],
    [-68.07548734644243, 46.981170472447374],
    [-69.17500995805074, 46.98158998130476],
    [-70.7598785138901, 44.87172183866657],
    [-74.84994741250935, 44.748084983808],
    [-77.62168256782745, 43.005725611950055],
    [-82.45987924104175, 41.41068867019324],
    [-83.38318501671864, 42.09979904377044],
    [-82.5905167831457, 45.06163491639556],
    [-84.83301910769038, 46.83552648258547],
    [-88.26350848510909, 48.143646480291835],
    [-90.06706251069104, 47.553445811024204],
    [-95.03745451438925, 48.9881557770297],
    [-98.45773319567587, 48.94699366043251],
    [-101.7018751401119, 48.98284560308372],
    [-108.43164852530356, 48.81973606668503],
    [-115.07339190755627, 48.93699058308441],
    [-121.82530604190744, 48.9830983403776],
    [-122.22085227110232, 48.63535795404536],
    [-124.59504332589562, 47.695726563030405],
    [-125.1803892906456, 35.26328285844432]
]

def create_usa_geometry():
    """Create an Earth Engine geometry object for the contiguous USA."""
    return ee.Geometry.Polygon([usa_coords])

def compute_area(feature):
    """Compute the area of a feature and set it as a property."""
    return feature.set({'area': feature.area()})

def compute_centroid(feature):
    """Compute the centroid coordinates of a feature and set them as properties."""
    centroid = feature.geometry().centroid().coordinates()
    return feature.set({
        'lon': centroid.get(0),
        'lat': centroid.get(1)
    })

def ee_featurecollection_to_gdf(fc):
    """Convert Earth Engine FeatureCollection to GeoPandas DataFrame."""
    features = fc.getInfo()['features']
    
    # Extract the geometry and properties from each feature
    geometries = []
    properties = []
    
    for feature in features:
        # Convert GEE geometry to Shapely geometry
        geom = feature['geometry']
        if geom['type'] == 'Polygon':
            geometry = Polygon(geom['coordinates'][0])
        else:
            # Handle other geometry types if needed
            continue
            
        geometries.append(geometry)
        properties.append(feature['properties'])
    
    # Create GeoDataFrame
    df = pd.DataFrame(properties)
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    
    # Convert area to numeric
    if 'area' in gdf.columns:
        gdf['area'] = pd.to_numeric(gdf['area'])
    
    return gdf

def get_daily_fires(year, min_size=1e7, region=None):
    """
    Get daily fire perimeters from the GlobFire database.
    
    Args:
        year (str): The year to get fires for
        min_size (float): Minimum fire size in square meters
        region (ee.Geometry, optional): Region to filter fires
    """
    if region is None:
        region = create_usa_geometry()
    
    collection_name = f'JRC/GWIS/GlobFire/v2/DailyPerimeters/{year}'
    
    try:
        polygons = (ee.FeatureCollection(collection_name)
                   .filterBounds(region))
        
        polygons = polygons.map(compute_area)
        polygons = (polygons
                   .filter(ee.Filter.gt('area', min_size))
                   .filter(ee.Filter.lt('area', 1e20)))
        
        polygons = polygons.map(compute_centroid)
        
        gdf = ee_featurecollection_to_gdf(polygons)
        
        if not gdf.empty:
            gdf['source'] = 'daily'
            # Convert IDate to datetime directly for each row
            gdf['date'] = pd.to_datetime(gdf['IDate'], unit='ms')
            # For daily perimeters, end_date is same as start date
            gdf['end_date'] = gdf['date']
        
        return gdf
        
    except ee.ee_exception.EEException as e:
        print(f"Error accessing daily collection for {year}: {str(e)}")
        return None

def get_final_fires(year, min_size=1e7, region=None):
    """
    Get final fire perimeters from the GlobFire database.
    
    Args:
        year (str): The year to get fires for
        min_size (float): Minimum fire size in square meters
        region (ee.Geometry, optional): Region to filter fires
    """
    if region is None:
        region = create_usa_geometry()
    
    start_date = ee.Date(f'{year}-01-01')
    end_date = ee.Date(f'{year}-12-31')
    
    try:
        polygons = (ee.FeatureCollection('JRC/GWIS/GlobFire/v2/FinalPerimeters')
                   .filter(ee.Filter.gt('IDate', start_date.millis()))
                   .filter(ee.Filter.lt('IDate', end_date.millis()))
                   .filterBounds(region))
        
        polygons = polygons.map(compute_area)
        polygons = (polygons
                   .filter(ee.Filter.gt('area', min_size))
                   .filter(ee.Filter.lt('area', 1e20)))
        
        polygons = polygons.map(compute_centroid)
        
        gdf = ee_featurecollection_to_gdf(polygons)
        
        if not gdf.empty:
            gdf['source'] = 'final'
            # Convert IDate and FDate to datetime for each row
            gdf['date'] = pd.to_datetime(gdf['IDate'], unit='ms')
            gdf['end_date'] = pd.to_datetime(gdf['FDate'], unit='ms')
        
        return gdf
        
    except ee.ee_exception.EEException as e:
        print(f"Error accessing final perimeters for {year}: {str(e)}")
        return None

def get_combined_fires(year, min_size=1e7, region=None):
    """
    Get both daily and final fire perimeters and combine them based on Id.
    
    Args:
        year (str): The year to get fires for
        min_size (float): Minimum fire size in square meters
        region (ee.Geometry, optional): Region to filter fires
    
    Returns:
        tuple: (combined_gdf, daily_gdf, final_gdf)
    """
    daily_gdf = get_daily_fires(year, min_size, region)
    final_gdf = get_final_fires(year, min_size, region)
    
    if daily_gdf is None and final_gdf is None:
        return None, None, None
    
    # Ensure we have dataframes to work with
    if daily_gdf is None:
        daily_gdf = gpd.GeoDataFrame()
    if final_gdf is None:
        final_gdf = gpd.GeoDataFrame()
    
    # Convert timestamps consistently
    for gdf in [daily_gdf, final_gdf]:
        if not gdf.empty:
            # Convert all timestamp fields to numeric if they aren't already
            for col in ['IDate', 'FDate']:
                if col in gdf.columns:
                    gdf[col] = pd.to_numeric(gdf[col])
            for col in ['FDate']:
                if col in gdf.columns:
                    gdf[col] = gdf['end_date']
    
    # Get unique fire IDs
    all_ids = pd.concat([
        daily_gdf['Id'] if not daily_gdf.empty else pd.Series(dtype=int),
        final_gdf['Id'] if not final_gdf.empty else pd.Series(dtype=int)
    ]).unique()
    
    combined_data = []
    
    for fire_id in all_ids:
        # Get daily perimeters for this fire
        daily_fire = daily_gdf[daily_gdf['Id'] == fire_id] if not daily_gdf.empty else None
        # Get final perimeter for this fire
        final_fire = final_gdf[final_gdf['Id'] == fire_id] if not final_gdf.empty else None
        
        if daily_fire is not None and not daily_fire.empty:
            # Add all daily perimeters
            combined_data.append(daily_fire)
        
        if final_fire is not None and not final_fire.empty:
            # Add final perimeter
            combined_data.append(final_fire)
    
    if not combined_data:
        return None, None, None
    
    # Combine all data
    combined_gdf = pd.concat(combined_data, ignore_index=True)
    
    # Sort by Id and date for consistency
    combined_gdf = combined_gdf.sort_values(['Id', 'date'])
    
    return combined_gdf, daily_gdf, final_gdf

def analyze_fires(gdf):
    """
    Perform basic analysis on fire perimeters.
    """
    if gdf is None or len(gdf) == 0:
        return None
    
    # Basic statistics
    stats = {
        'total_fires': len(gdf),
        'unique_fires': gdf['Id'].nunique(),
        'total_area_km2': gdf['area'].sum() / 1e6,
        'mean_area_km2': gdf['area'].mean() / 1e6,
        'max_area_km2': gdf['area'].max() / 1e6,
        'date_range': f"{gdf['date'].min()} to {gdf['end_date'].max()}"
    }
    
    # Add source-specific counts
    if 'source' in gdf.columns:
        source_counts = gdf['source'].value_counts()
        for source in source_counts.index:
            stats[f'{source}_fires'] = source_counts[source]
            
        # Add counts of fires with both daily and final perimeters
        fires_with_both = (gdf.groupby('Id')['source']
                          .nunique()
                          .where(lambda x: x > 1)
                          .count())
        stats['fires_with_both_perims'] = fires_with_both
    
    return stats

# Example usage
YEAR = "2020"
MIN_SIZE = 1e7  # 10 square kilometers

# Get both daily and final perimeters
combined_gdf, daily_gdf, final_gdf = get_combined_fires(YEAR, MIN_SIZE)

if combined_gdf is not None:
    print(f"\nAnalysis Results for {YEAR}:")
    
    print("\nCombined Perimeters:")
    combined_stats = analyze_fires(combined_gdf)
    for key, value in combined_stats.items():
        print(f"{key}: {value}")
    
    if daily_gdf is not None:
        print("\nDaily Perimeters:")
        daily_stats = analyze_fires(daily_gdf)
        for key, value in daily_stats.items():
            print(f"{key}: {value}")
    
    if final_gdf is not None:
        print("\nFinal Perimeters:")
        final_stats = analyze_fires(final_gdf)
        for key, value in final_stats.items():
            print(f"{key}: {value}")
    
    # Temporal distribution
    print("\nFires by month:")
    monthly_counts = combined_gdf.groupby([combined_gdf['date'].dt.month, 'source']).size().unstack(fill_value=0)
    print(monthly_counts)

# drop everything that does not have at least 2 Id in combined_gdf
combined_gdf_reduced = combined_gdf[combined_gdf['Id'].isin(combined_gdf['Id'].value_counts()[combined_gdf['Id'].value_counts() > 1].index)]# save to geojson
combined_gdf_reduced.to_file(f"data/perims/combined_fires_{YEAR}.geojson", driver="GeoJSON")