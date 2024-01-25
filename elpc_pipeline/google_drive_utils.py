'''
Utils for interacting with Google Drive 

- Assumes connecting via a service account rather than user auth via web

Currently the connaction is done using oauth2client, which is deprecated. It works, but
may need to be replaced with google-auth. One attempt was made at this, but was not successful.

TODOs:
- Add a retry decorator that can automatically reconnect if timeout occurs
- Add support for shared/team drives:  https://stackoverflow.com/a/57577212

'''

import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe

def connect(service_acct_json):
    gauth = GoogleAuth()
    scope = ["https://www.googleapis.com/auth/drive"]
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(service_acct_json, scope)
    drive = GoogleDrive(gauth)
    return drive

def get_obj_id_by_name(drive, obj_name, parent_id=None, exclude_trashed=False):
    query = f"title='{obj_name}'"
    if exclude_trashed:
        query += " and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    file_list = drive.ListFile({'q': query}).GetList()
    if len(file_list) == 0:
        raise ValueError("No matching files found!")
    elif len(file_list) > 1:
        raise ValueError("Multiple matching files found!")
    return file_list[0]['id']

def mkdir(drive, folder_name, parent_folder_id=None):
    new_file = drive.CreateFile({
        'title': folder_name, 
        'mimeType': 'application/vnd.google-apps.folder'
        })
    if parent_folder_id:
        new_file['parents'] = [{
            'kind': 'drive#parentReference', 
            'id': parent_folder_id
            }]
    new_file.Upload()
    new_file.FetchMetadata()
    return new_file

def upload_from_file(drive, local_file, parent_folder_id=None, public_viewable=False):
    new_file = drive.CreateFile()
    new_file.SetContentFile(local_file)
    new_file['title'] = os.path.split(local_file)[1]
    if parent_folder_id:
        new_file['parents'] = [{
            'kind': 'drive#parentReference', 
            'id': parent_folder_id
            }]
    new_file.Upload()  
    new_file.FetchMetadata()
    if public_viewable:
        new_file.InsertPermission({
            'type': 'anyone',
            'value': 'anyone',
            'role': 'reader'
        })
    return new_file

def gmaps_url(lat, lon, zoom=None):
    # expects lat/lon in decimal degrees
    url = f'http://maps.google.com/maps?q={lat},{lon}'
    if zoom:
        url += f'&z={zoom}'
    return url

def bing_maps_url(lat, lon, zoom=None):
    # expects lat/lon in decimal degrees
    url = f'https://www.bing.com/maps/?cp={lat}~{lon}&lvl=16.0&style=h'
    return url

# TODO: check if this comes back as one of the elements of the drive file object...
def image_embed_url(image_id):
    return f'https://drive.google.com/uc?export=view&id={image_id}'

def append_to_gsheet(service_acct_json, new_df, sheet_key, tab=0, check_header=False):
    gc = gspread.service_account(filename=service_acct_json)
    gsheet = gc.open_by_key(sheet_key)
    worksheet = gsheet.get_worksheet(tab)

    df = get_as_dataframe(worksheet)
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c]).dropna(how="all")
    curr_rows = df.shape[0]

    if check_header:
        # additional columns beyond the new data are ok, but initial column names must all match with expected format
        new_cols = list(new_df.columns)
        sheet_cols = list(df.columns)
        if not sheet_cols[:len(new_cols)] == new_cols:
            raise ValueError("GSheets Append Aborted -- Columns Don't Match!")

    # need to add 2 to current rows (header is in row 1)
    set_with_dataframe(worksheet, new_df, row=curr_rows+2, col=1, include_index=False, include_column_header=False)

def df_from_gsheet(service_acct_json, sheet_key, tab_gid=0):
    gc = gspread.service_account(filename=service_acct_json)
    gsheet = gc.open_by_key(sheet_key)
    worksheet = gsheet.get_worksheet_by_id(tab_gid)

    df = get_as_dataframe(worksheet)
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c]).dropna(how="all")
    return df
