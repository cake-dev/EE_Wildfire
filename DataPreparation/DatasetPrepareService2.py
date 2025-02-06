import datetime
import os

import ee
import imageio

class DatasetPrepareService2:
    
    def __init__(self, location, config):
        """Class that handles downloading data associated with the given location and time period from Google Earth Engine.

        Args:
            location: Location for which to download data. Must be a key in the config file.
            config: Config file containing the location and time period for which to download data, 
            as well as the size of the rectangular area to extract.
        """
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

    def cast_to_uint8(self, image):
        return image.multiply(512).uint8()
        
    def prepare_daily_image(self, date_of_interest:str, time_stamp_start:str="00:00", time_stamp_end:str="23:59"):
        """Prepare daily image from GEE.

        Args:
            date_of_interest (str): Date for which we want to download data.
            time_stamp_start (str, optional): String representation of start of day time. Defaults to "00:00".
            time_stamp_end (str, optional): String representation of end of day time. Defaults to "23:59".

        Returns:
            ImageCollection containing one image with all desired features for the given day.
        """
        from .satellites.FirePred import FirePred
        satellite_client = FirePred()
        img_collection = satellite_client.compute_daily_features(date_of_interest + 'T' + time_stamp_start,
                                                               date_of_interest + 'T' + time_stamp_end,
                                                               self.geometry)        
        return img_collection

    def download_image_to_drive(self, image_collection, index:str, utm_zone:str):
        """Export the given images to Google Drive.

        Args:
            image_collection: Image collection to export
            index (str): Date identifier for the file
            utm_zone (str): UTM zone for projection
        """
        if "year" in self.config:
            folder = f"WildfireSpreadTS_{self.config['year']}"
            filename = f"{self.location}/{index}"
        else:
            folder = "WildfireSpreadTS"
            filename = f"{self.location}/{index}"

        img = image_collection.max().toFloat()
        image_task = ee.batch.Export.image.toDrive(
            image=img,
            description='Image Export',
            folder=folder,
            fileNamePrefix=filename,
            scale=self.scale_dict.get("FirePred"),
            crs='EPSG:' + utm_zone,
            maxPixels=1e13,
            region=self.geometry.toGeoJSON()['coordinates'],
        )
        print(f'Starting image task (id: {image_task.id}) for {filename}')
        image_task.start()
        
    def extract_dataset_from_gee_to_drive(self, utm_zone:str, n_buffer_days:int=0):
        """Iterate over the time period and download the data for each day to Google Drive.

        Args:
            utm_zone (str): UTM zone for projection
            n_buffer_days (int, optional): Number of days before and after the fire dates 
            for which we also want to collect data. Defaults to 0.

        Raises:
            RuntimeError: If more than one feature is found in image collection
        """
        buffer_days = datetime.timedelta(days=n_buffer_days)
        time_dif = self.end_time - self.start_time + 2 * buffer_days + datetime.timedelta(days=1)

        for i in range(time_dif.days):
            date_of_interest = str(self.start_time - buffer_days + datetime.timedelta(days=i))

            img_collection = self.prepare_daily_image(date_of_interest=date_of_interest)

            n_images = len(img_collection.getInfo().get("features"))
            if n_images > 1:
                raise RuntimeError(f"Found {n_images} features in img_collection returned by prepare_daily_image. "
                                 f"Should have been exactly 1.")
            max_img = img_collection.max()
            if len(max_img.getInfo().get('bands')) != 0:
                self.download_image_to_drive(img_collection, date_of_interest, utm_zone)