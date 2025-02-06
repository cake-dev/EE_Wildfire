import ee
import yaml
import tqdm
import geemap
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

from DataPreparation.DatasetPrepareService import DatasetPrepareService

if __name__ == '__main__':
    # Load config file
    with open("config/us_fire_2017_1e7.yml", "r", encoding="utf8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize Earth Engine
    service_account = 'bova-ee@ee-earthdata.iam.gserviceaccount.com'
    credentials = ee.ServiceAccountCredentials(service_account, 'gcloud_key.json')
    ee.Initialize(credentials)

    # authenticate to Google Drive (of the Service account)
    gauth = GoogleAuth()
    scopes = ['https://www.googleapis.com/auth/drive']
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name("gcloud_key.json", scopes=scopes)

    drive = GoogleDrive(gauth)

    # Number of buffer days
    N_BUFFER_DAYS = 4

    # Extract fire names from config
    fire_names = list(config.keys())
    for non_fire_key in ["output_bucket", "rectangular_size", "year"]:
        fire_names.remove(non_fire_key)
    locations = fire_names

    # Track any failures
    failed_locations = []

    # Process each location
    for location in tqdm.tqdm(locations):
        print(f"Failed locations so far: {failed_locations}")
        dataset_pre = DatasetPrepareService(location=location, config=config)
        print("Current Location:" + location)

        try:
            # Now using Google Drive instead of Cloud Storage
            dataset_pre.extract_dataset_from_gee_to_drive('32610', n_buffer_days=N_BUFFER_DAYS)
        except Exception as e:
            print(f"Failed on {location}: {str(e)}")
            failed_locations.append(location)