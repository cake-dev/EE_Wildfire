import os
import shutil
from datetime import datetime

def organize_files(source_dir, target_base_dir):
    """
    Organizes files from source directory into structured format:
    wfspreads_org/YYYY/fire_NUMBER/YYYY-MM-DD.tif
    """
    # Create base directory if it doesn't exist
    os.makedirs(target_base_dir, exist_ok=True)

    # Process each file in the source directory
    for filename in os.listdir(source_dir):
        if filename.startswith('Image_Export_fire_'):
            # Extract date from filename
            # Example: Image_Export_fire_20701045_2017-06-14.tif
            parts = filename.split('_')
            fire_number = parts[3]
            date_str = parts[4].replace('.tif', '')
            
            # Convert date to desired format
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            new_date = date_obj.strftime('%Y-%m-%d')
            year = date_obj.strftime('%Y')
            
            # Create directory structure
            fire_dir = os.path.join(target_base_dir, year, f'fire_{fire_number}')
            os.makedirs(fire_dir, exist_ok=True)
            
            # Create new filename
            new_filename = f'{new_date}.tif'
            
            # Full paths
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(fire_dir, new_filename)
            
            # Move and rename file
            shutil.move(source_path, target_path)
            print(f'Moved {filename} to {target_path}')

if __name__ == '__main__':
    # Example usage
    source_directory = 'data/download'
    target_directory = 'data/wfspreadts_orig'
    
    organize_files(source_directory, target_directory)