import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from pathlib import Path
from tqdm import tqdm

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class DriveDownloader:
    def __init__(self):
        self.service = self._get_drive_service()
        
    def _get_drive_service(self):
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        return build('drive', 'v3', credentials=creds)

    def _get_folder_id(self, folder_path: str):
        """Get folder ID by traversing the path."""
        parts = folder_path.strip('/').split('/')
        parent_id = 'root'
        
        for part in parts:
            query = f"name='{part}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder'"
            results = self.service.files().list(q=query, spaces='drive').execute()
            items = results.get('files', [])
            
            if not items:
                raise ValueError(f"Folder '{part}' not found in path '{folder_path}'")
            parent_id = items[0]['id']
            
        return parent_id

    def download_folder(self, drive_folder_path: str, local_path: str):
        """Download all files from the specified Drive folder."""
        try:
            folder_id = self._get_folder_id(drive_folder_path)
            output_dir = Path(local_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all files in the folder with pagination
            files = []
            page_token = None
            while True:
                results = self.service.files().list(
                    q=f"'{folder_id}' in parents",
                    spaces='drive',
                    pageSize=1000,
                    fields='nextPageToken, files(id, name)',
                    pageToken=page_token
                ).execute()
                files.extend(results.get('files', []))
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
            
            print(f"Found {len(files)} files")
            
            for file in tqdm(files, desc="Downloading files"):
                request = self.service.files().get_media(fileId=file['id'])
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                
                output_path = output_dir / file['name']
                with open(output_path, 'wb') as f:
                    f.write(fh.getvalue())
                    
        except Exception as e:
            print(f"Error downloading folder: {str(e)}")
            raise

def main():
    drive_folder = input("Enter Drive folder path (e.g., EarthEngine_WildfireSpreadTS): ")
    local_folder = input("Enter local download path: ")
    
    downloader = DriveDownloader()
    downloader.download_folder(drive_folder, local_folder)
    print("\nDownload completed!")

if __name__ == '__main__':
    main()