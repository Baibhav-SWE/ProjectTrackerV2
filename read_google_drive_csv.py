from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pandas as pd
import io
import os.path
import pickle

def authenticate_google_drive():
    """Authenticate with Google Drive API."""
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = None
    
    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        try:
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        except Exception:
            if os.path.exists('token.pickle'):
                os.remove('token.pickle')
            creds = None
    
    # If credentials are not valid or don't exist, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print("Error refreshing credentials:", str(e))
                print("\nPlease ensure you have:")
                print("1. Set up the OAuth consent screen in Google Cloud Console")
                print("2. Added yourself as a test user")
                print("3. Downloaded the latest credentials.json file")
                return None
        else:
            try:
                if not os.path.exists('credentials.json'):
                    print("\nError: credentials.json file not found!")
                    print("Please download it from Google Cloud Console and place it in the same directory as this script.")
                    return None
                    
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)
            except Exception as e:
                print("\nAuthentication Error:", str(e))
                print("\nPlease ensure you have:")
                print("1. Set up the OAuth consent screen in Google Cloud Console")
                print("2. Added yourself as a test user")
                print("3. Downloaded the latest credentials.json file")
                return None
    
    return creds

def read_csv_from_drive(file_id):
    """
    Read a CSV file from Google Drive and return it as a pandas DataFrame.
    
    Args:
        file_id (str): The ID of the file in Google Drive
                      (can be found in the sharing URL)
    
    Returns:
        pandas.DataFrame: The contents of the CSV file
    """
    try:
        # Authenticate and build the Drive API service
        creds = authenticate_google_drive()
        if creds is None:
            return None
            
        service = build('drive', 'v3', credentials=creds)
        
        # Download the file content
        request = service.files().get_media(fileId=file_id)
        file_handle = io.BytesIO()
        downloader = MediaIoBaseDownload(file_handle, request)
        done = False
        
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download Progress: {int(status.progress() * 100)}%")
        
        # Reset the file handle position to the beginning
        file_handle.seek(0)
        
        # Read the CSV content into a pandas DataFrame
        df = pd.read_csv(file_handle)
        return df
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        if "access_denied" in str(e).lower():
            print("\nAccess Denied. Please ensure:")
            print("1. You have been added as a test user in the OAuth consent screen")
            print("2. The file ID is correct and the file is accessible to your Google account")
            print("3. You have enabled the Google Drive API in your Google Cloud Console")
        return None

def main():
    # Example usage
    # Replace with your Google Drive CSV file ID
    file_id = '18vdMqrm2Mp0Efbn6Fha5ruYmd-mbXxd6'
    
    # Read the CSV file
    df = read_csv_from_drive(file_id)
    
    if df is not None:
        print("\nFirst few rows of the CSV file:")
        print(df.head())
        print("\nDataFrame info:")
        print(df.info())

if __name__ == '__main__':
    main()