import datetime
import ee
import geemap
import yaml
from tqdm import tqdm
import sys
from pathlib import Path
import time

# Add the parent directory to the Python path to enable imports
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from DataPreparation.satellites.FirePred import FirePred

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
        self.total_tasks = 0

        # Set the area to extract as an image
        self.rectangular_size = self.config.get('rectangular_size')
        self.geometry = ee.Geometry.Rectangle(
            [self.longitude - self.rectangular_size, self.latitude - self.rectangular_size,
             self.longitude + self.rectangular_size, self.latitude + self.rectangular_size])

        self.scale_dict = {"FirePred": 375}
        
    def prepare_daily_image(self, date_of_interest:str, time_stamp_start:str="00:00", time_stamp_end:str="23:59"):
        """Prepare daily image from GEE."""
        self.total_tasks += 1
        if self.total_tasks > 2500:
            active_tasks = str(ee.batch.Task.list()).count('READY')
            while active_tasks > 2000:
                time.sleep(60)
                active_tasks = str(ee.batch.Task.list()).count('READY')
        satellite_client = FirePred()
        img_collection = satellite_client.compute_daily_features(date_of_interest + 'T' + time_stamp_start,
                                                               date_of_interest + 'T' + time_stamp_end,
                                                               self.geometry)        
        return img_collection

    def download_image_to_drive(self, image_collection, index:str, utm_zone:str):
        """Export the given images to Google Drive using geemap."""
        if "year" in self.config:
            folder = f"EarthEngine_WildfireSpreadTS_{self.config['year']}"
            filename = f"{self.location}/{index}"
        else:
            folder = "EarthEngine_WildfireSpreadTS"
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
                # wait to avoid rate limiting
                # time.sleep(1)

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

def main():
    # Load config file
    YEAR = input("Enter the year: ")
    with open(f"config/us_fire_{YEAR}_1e7.yml", "r", encoding="utf8") as f:
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
    main()