import ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, mapping
from datetime import datetime, timedelta
import yaml
import geemap
from tqdm import tqdm
import sys
from pathlib import Path
import math

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

############################################################################################################################################################


def create_fire_config(geojson_path, output_path, year):
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
# create_fire_config('data/perims/combined_fires_2017.geojson', 'WildfireSpreadTSCreateDataset/config/us_fire_2017_1e7.yml', 2017)

############################################################################################################################################################

# Add the parent directory to the Python path to enable imports
# root_dir = str(Path(__file__).parent.parent)
# if root_dir not in sys.path:
#     sys.path.append(root_dir)

class FirePred:
    def __init__(self):
        """_summary_ This class describes which data to extract how from Google Earth Engine. 
        The init defines the different source data products to use. 
        """
        self.name = "FirePred"
        # Digital elevation model
        self.srtm = ee.Image("USGS/SRTMGL1_003")
        # Land cover
        self.landcover = ee.ImageCollection("MODIS/061/MCD12Q1")
        # Weather data (GRIDMET)
        self.weather = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
        # Weather forecast data (GFS)
        self.weather_forecast = ee.ImageCollection('NOAA/GFS0P25')
        # Drought data (GRIDMET)
        self.drought = ee.ImageCollection("GRIDMET/DROUGHT")
        # VIIRS surface reflectance
        self.viirs = ee.ImageCollection('NOAA/VIIRS/001/VNP09GA')
        # VIIRS active fire product
        self.viirs_af = ee.FeatureCollection('projects/grand-drive-285514/assets/afall')
        # VIIRS vegetation index
        self.viirs_veg_idx = ee.ImageCollection("NOAA/VIIRS/001/VNP13A1")

    def compute_daily_features(self, start_time:str, end_time:str, geometry:ee.Geometry):
        """_summary_ Compute the daily features in Google Earth Engine.

        Args:
            start_time (str): _description_
            end_time (str): _description_
            geometry (ee.Geometry): _description_

        Returns:
            ee.ImageCollection: _description_ ImageCollection containing one image, 
            with all desired features for the given day, inside the given geometry.
        """


        # Time objects we need later. We add "000" to timestamps, because GEE has timestamps with miliseconds,
        # but datetime doesn't by default
        today_string = start_time[:-6].replace("-", "")
        today = datetime.datetime.strptime(start_time[:-6], '%Y-%m-%d')
        today_timestamp = int(datetime.datetime.timestamp(today)) * 1000

        # Weather Data
        # Median is used to turn ee.ImageCollection into a single ee.Image.
        # Each ImageCollection should only contain a single image at this point.
        weather = self.weather.filterDate(start_time, end_time).filterBounds(geometry)
        precipitation = weather.select('pr').median().rename("total precipitation")
        wind_direction = weather.select('th').median().rename("wind direction")
        temperature_min = weather.select('tmmn').median().rename("minimum temperature")
        temperature_max = weather.select('tmmx').median().rename("maximum temperature")
        energy_release_component = weather.select('erc').median().rename("energy release component")
        specific_humidity = weather.select('sph').median().rename("specific humidity")
        wind_velocity = weather.select('vs').median().rename("wind speed")

        # Take forecasts made at midnight (00), and that tell us something about the hours between 01 and 24.
        # Important: The forecasts at 00 contain six features instead of nine, like all others.
        weather_forecast = self.weather_forecast.filter(
            ee.Filter.gte("system:index", today_string + "00F01")).filter(
            ee.Filter.lte("system:index", today_string + "00F24")
        ).filterBounds(geometry)
        forecast_temperature = weather_forecast.select("temperature_2m_above_ground").mean().rename(
            "forecast temperature")
        forecast_specific_humidity = weather_forecast.select("specific_humidity_2m_above_ground").mean().rename(
            "forecast specific humidity")
        forecast_u_wind = weather_forecast.select("u_component_of_wind_10m_above_ground").mean()
        forecast_v_wind = weather_forecast.select("v_component_of_wind_10m_above_ground").mean()

        # Transform from u/v to direction and speed, to align with GRIDMET and DEM data
        forecast_wind_speed = forecast_u_wind.multiply(forecast_u_wind).add(
            forecast_v_wind.multiply(forecast_v_wind)).sqrt().rename("forecast wind speed")
        forecast_wind_direction = forecast_v_wind.divide(forecast_u_wind).atan()
        forecast_wind_direction = forecast_wind_direction.divide(2 * math.pi).multiply(360).rename(
            "forecast wind direction")

        # Rain forecasts were changed: From rain within the one-hour interval to cumulative rain during the day so far
        forecast_rain_change_date = datetime.datetime.strptime("2019-11-07T06:00:00", '%Y-%m-%dT%H:%M:%S')
        forecast_rain = weather_forecast.select("total_precipitation_surface")
        if today <= forecast_rain_change_date:
            forecast_rain = forecast_rain.reduce(ee.Reducer.sum())
        else:
            forecast_rain = forecast_rain.reduce(ee.Reducer.last())
        forecast_rain.rename("forecast total precipitation")
        # Elevation Data
        elevation = self.srtm.select('elevation')
        slope = ee.Terrain.slope(elevation)
        aspect = ee.Terrain.aspect(elevation)

        # Drought Data
        # Only available every fifth day, but we can find the valid entry via time_start and time_end
        drought_index = self.drought \
            .filter(ee.Filter.lte("system:time_start", today_timestamp)) \
            .filter(ee.Filter.gte("system:time_end", today_timestamp)) \
            .select('pdsi').median()
        igbp_land_cover = self.landcover.filterDate(start_time[:4] + '-01-01', start_time[:4] + '-12-31').filterBounds(
            geometry).select('LC_Type1').median()

        # Turn acq_time (String) into acq_hour (int)
        def add_acq_hour(feature):
            acq_time_str = ee.String(feature.get("acq_time"))
            acq_time_int = ee.Number.parse(acq_time_str)
            return feature.set({"acq_hour": acq_time_int})

        # VIIRS IMG and AF product
        viirs_img = self.viirs.filterDate(start_time, end_time).filterBounds(geometry).select(
            ['M11', 'I2', 'I1']).median()
        viirs_veg_idc = self.viirs_veg_idx.filterDate((
                datetime.datetime.strptime(end_time[:-6], '%Y-%m-%d') + datetime.timedelta(-15)).strftime(
            '%Y-%m-%d'), end_time).filterBounds(geometry).select(['NDVI', 'EVI2']).reduce(
            ee.Reducer.last())

        # VIIRS AF consists only of points, so we need to turn them into a raster image.
        # We also filter out low confidence detections, since they are most likely false positives. 
        viirs_af_img = self.viirs_af.map(add_acq_hour).filterBounds(geometry) \
            .filter(ee.Filter.gte('acq_date', start_time[:-6])) \
            .filter(ee.Filter.lt('acq_date', (
                datetime.datetime.strptime(end_time[:-6], '%Y-%m-%d') + datetime.timedelta(1)).strftime(
            '%Y-%m-%d'))) \
            .filter(ee.Filter.neq('confidence', 'l')).map(self.get_buffer) \
            .reduceToImage(['acq_hour'], ee.Reducer.last()) \
            .rename(['active fire'])

        return ee.ImageCollection(ee.Image(
            [viirs_img, viirs_veg_idc, precipitation, wind_velocity, wind_direction, temperature_min, temperature_max,
             energy_release_component, specific_humidity, slope, aspect,
             elevation, drought_index, igbp_land_cover,
             forecast_rain, forecast_wind_speed, forecast_wind_direction, forecast_temperature,
             forecast_specific_humidity,
             viirs_af_img]))

    def get_buffer(self, feature):
        return feature.buffer(375 / 2).bounds()


# from DataPreparation.satellites.FirePred import FirePred

class DatasetPrepareService:
    def __init__(self, location, config):
        """Class that handles downloading data associated with the given location and time period from Google Earth Engine."""
        self.config = config
        self.location = location
        self.rectangular_size = self.config.get('rectangular_size')
        self.latitude = self.config.get(self.location).get('latitude')
        self.longitude = self.config.get(self.location).get('longitude')
        self.start_time = self.config.get(location).get('start')
        self.end_time = self.config.get(location).get('end')

        # Set the area to extract as an image
        self.rectangular_size = self.config.get('rectangular_size')
        self.geometry = ee.Geometry.Rectangle(
            [self.longitude - self.rectangular_size, self.latitude - self.rectangular_size,
             self.longitude + self.rectangular_size, self.latitude + self.rectangular_size])

        self.scale_dict = {"FirePred": 375}
        
    def prepare_daily_image(self, date_of_interest:str, time_stamp_start:str="00:00", time_stamp_end:str="23:59"):
        """Prepare daily image from GEE."""
        satellite_client = FirePred()
        img_collection = satellite_client.compute_daily_features(date_of_interest + 'T' + time_stamp_start,
                                                               date_of_interest + 'T' + time_stamp_end,
                                                               self.geometry)        
        return img_collection

    def download_image_to_drive(self, image_collection, index:str, utm_zone:str):
        """Export the given images to Google Drive using geemap."""
        if "year" in self.config:
            folder = f"EarthEngine/WildfireSpreadTS_{self.config['year']}"
            filename = f"{self.location}/{index}"
        else:
            folder = "EarthEngine/WildfireSpreadTS"
            filename = f"{self.location}/{index}"

        img = image_collection.max().toFloat()
        
        # Use geemap's export function
        try:
            geemap.ee_export_image_to_drive(
                image=img,
                description=f'Image_Export_{self.location}_{index}',
                folder=folder,
                region=self.geometry.toGeoJSON()['coordinates'],
                scale=self.scale_dict.get("FirePred"),
                crs=f'EPSG:{utm_zone}',
                maxPixels=1e13
            )
            print(f"Successfully queued export for {filename}")
        except Exception as e:
            print(f"Export failed for {filename}: {str(e)}")
            raise
        
    def extract_dataset_from_gee_to_drive(self, utm_zone:str, n_buffer_days:int=0):
        """Iterate over the time period and download the data for each day to Google Drive."""
        buffer_days = datetime.timedelta(days=n_buffer_days)
        time_dif = self.end_time - self.start_time + 2 * buffer_days + datetime.timedelta(days=1)

        for i in range(time_dif.days):
            date_of_interest = str(self.start_time - buffer_days + datetime.timedelta(days=i))
            print(f"Processing date: {date_of_interest}")

            try:
                img_collection = self.prepare_daily_image(date_of_interest=date_of_interest)

                n_images = len(img_collection.getInfo().get("features"))
                if n_images > 1:
                    raise RuntimeError(f"Found {n_images} features in img_collection returned by prepare_daily_image. "
                                     f"Should have been exactly 1.")
                max_img = img_collection.max()
                if len(max_img.getInfo().get('bands')) != 0:
                    self.download_image_to_drive(img_collection, date_of_interest, utm_zone)
            except Exception as e:
                print(f"Failed processing {date_of_interest}: {str(e)}")
                raise

def main(year):
    # Load config file
    with open(f"config/us_fire_{year}_1e7.yml", "r", encoding="utf8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize Earth Engine with your personal account
    geemap.ee_initialize()

    # Extract fire names from config
    fire_names = list(config.keys())
    for non_fire_key in ["output_bucket", "rectangular_size", "year"]:
        fire_names.remove(non_fire_key)
    locations = fire_names

    # Track any failures
    failed_locations = []

    # Process each location
    for location in tqdm(locations):
        print(f"\nFailed locations so far: {failed_locations}")
        print(f"Current Location: {location}")
        
        dataset_pre = DatasetPrepareService(location=location, config=config)

        try:
            dataset_pre.extract_dataset_from_gee_to_drive('32610', n_buffer_days=4)
        except Exception as e:
            print(f"Failed on {location}: {str(e)}")
            failed_locations.append(location)
            continue

    if failed_locations:
        print("\nFailed locations:")
        for loc in failed_locations:
            print(f"- {loc}")
    else:
        print("\nAll locations processed successfully!")

if __name__ == '__main__':
    # Example usage
    YEAR = input("Enter the year to analyze: ")
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
        # save to geojson
        # drop everything that does not have at least 2 Id in combined_gdf
        combined_gdf_reduced = combined_gdf[combined_gdf['Id'].isin(combined_gdf['Id'].value_counts()[combined_gdf['Id'].value_counts() > 1].index)]
        combined_gdf_reduced.to_file(f"data/perims/combined_fires_{YEAR}.geojson", driver="GeoJSON")
    create_fire_config(f'data/perims/combined_fires_{YEAR}.geojson', f'WildfireSpreadTSCreateDataset/config/us_fire_{YEAR}_1e7.yml', YEAR)
    main(YEAR)


############################################################################################################################################################
