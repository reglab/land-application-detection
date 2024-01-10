import os
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import numpy as np
import yaml
import ohio.ext.pandas
import sshtunnel
import geopandas as gpd

from add_predictions_to_sql import db_connect

import google_drive_utils as gd
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.patches as patches

import requests
from PIL import Image
from io import BytesIO

import logging
logger = logging.getLogger(__name__)

def rect_intersect(r1, r2): 
    """Return True iff given rectangles intersect. 
    rectangles defined as [left (min long), right (max long), bottom (min lat), top (max lat)]
    Helper function to check if detections are duplicates/were sent before"""

    [x1_left, x1_right, bottom1, top1] = r1 
    [x2_left, x2_right, bottom2, top2] = r2

    # check if either is to the left of the other 
    if (x1_right < x2_left) or (x2_right < x1_left): 
        return False

    # check if either is below the other 
    if top1 < bottom2 or top2 < bottom1: 
        return False 

    return True 

def row_coords(detection):
    '''Helper function to get the relevant coordinates from the dataframe'''

    if len(detection.shape) > 1:
        raise ValueError('row_coords expects exactly 1 detection event')
    return [
        detection['lon_min'],
        detection['lon_max'],
        detection['lat_min'],
        detection['lat_max']
    ]

def load_events(config, conn, db_schema, run_id=None, run_nickname=None):
    '''
    If run ID or run nickname is not specified, defaults to loading detections from the most recent run
    '''
    if run_id is None and run_nickname == "":
        run_id = conn.execute(f'''SELECT MAX(run_id) FROM {db_schema}.pipeline_runs 
                              WHERE run_timestamp = (SELECT MAX(run_timestamp) FROM {db_schema}.pipeline_runs)''').fetchall()[0][0]
    if run_id is None and run_nickname != "":
        run_id = conn.execute(f"SELECT MAX(run_id) FROM {db_schema}.pipeline_runs WHERE run_nickname = '{run_nickname}'").fetchall()[0][0]

    events = pd.read_sql(
        f"SELECT * FROM {db_schema}.detections WHERE run_id = '{run_id}'",
        conn)

    logging.info(f"{len(events)} detected events loaded from database for post-prediction.")
    logging.debug(f"Head of loaded events df: \n{events.head(5)}")

    return events


def filter_prev_seen(config, conn, db_schema, all_events):
    """
    Filters out detections previously sent to ELPC verifiers.
    """
    filter_days = config['prev_seen']['filter_days'] # how many days to look back at sent events for 

    curr_detection = all_events['detection_timestamp'].max().strftime('%Y%m%d %H:%M:%S')
    prev_sent = pd.read_sql(
        f"""
            SELECT d.*
            FROM {db_schema}.sent_to_elpc s
            JOIN {db_schema}.detections d
            USING(detection_id)
            WHERE s.detection_timestamp >= '{curr_detection}'::TIMESTAMP - INTERVAL '{filter_days} days'
        """,
        conn
    )

    logging.debug(f"Loaded {len(prev_sent)} previously sent events to compare for filtering")
    logging.debug(f"Previously sent events head: \n{prev_sent.head()}")

    new_events = []
    for _, ev in all_events.iterrows():
        filtered = False
        for _, prev in prev_sent.iterrows():
            if rect_intersect(row_coords(ev), row_coords(prev)):
                filtered=True
                break
        if not filtered:
            new_events.append(ev)

    logging.info(f"Filtered out {len(all_events) - len(new_events)} events that were previously sent to ELPC verifiers.")
    logging.info(f"Dataset now has {len(new_events)} events, after filtering out previously sent events.")    

    return pd.DataFrame(new_events)

def filter_duplicate_detections(config, all_events):
    """
    Checks to see if detected events intersect and removes the one that is smaller in area. This is to remove 
    repeat detections that may come from the images overlapping during the same detection run
    """

    distinct_events = pd.DataFrame(columns=all_events.columns)
    for i, ev1 in all_events.iterrows():
        duplicate_events = [ev1]
        top_score = ev1['score']
        for j, ev2 in all_events.iterrows():
            if i == j:
                continue
            #we have to make all the comparisons each time because the biggest entry needs to be 
            # preserved
            if rect_intersect(row_coords(ev1), row_coords(ev2)):
                top_score = max(top_score, ev2['score'])
                duplicate_events.append(ev2)
        
        duplicate_events = pd.DataFrame(duplicate_events)
        biggest = duplicate_events.sort_values('est_size_acres', ascending=False).iloc[0]
        biggest['score'] = top_score
        if not (distinct_events['detection_id'] == biggest['detection_id']).any():
            distinct_events = distinct_events.append(biggest)
            # Later: figure out why this isn't working
            #distinct_events = pd.concat([distinct_events, biggest.to_frame().transpose()], ignore_index=True)
    
    logging.info(f"Filtered out {len(all_events) - len(distinct_events)} events that were duplicate/repeat detections.")
    logging.info(f"Dataset now has {len(distinct_events)} events, after filtering out duplicate/repeat detections.")    

    return distinct_events



def meters_to_miles(dist):
  return dist * 0.000621371


def round_df_cols(df, round_config):
    for prec, cols in round_config.items():
        for col in cols:
            df[col] = df[col].round(prec)
            if prec == 0:
                df[col] = df[col].astype(int)
    return df

def make_dist_df(config, selected_events):
    selected_geos = gpd.GeoDataFrame(
            selected_events[['detection_id', 'lat_center', 'lon_center']], 
            geometry=gpd.points_from_xy(selected_events['lon_center'], selected_events['lat_center']), 
            crs='EPSG:4326'
        ).set_index('detection_id')
    selected_geos = selected_geos.to_crs(config['verifiers']['crs_proj'])

    verifier_df = gd.df_from_gsheet(
        config['google_drive']['creds_json'],
        config['verifiers']['gsheet_key'],
        tab_gid=config['verifiers']['gsheet_tab_gid']
    )

    logging.info(f"Successfully read in verifiers info, there are {len(verifier_df)} verifiers.")
    logging.debug(f"Head of verifiers df: {verifier_df.head()}")

    verifier_df = verifier_df.loc[verifier_df['active'] == 'Y', ]
    verifier_df['address_zip'] = verifier_df['address_zip'].astype(int).astype(str).str.zfill(5)
    verifier_geos = gpd.GeoDataFrame(
            verifier_df, 
            geometry=gpd.points_from_xy(verifier_df['address_lon'], verifier_df['address_lat']), 
            crs='EPSG:4326'
        )
    verifier_geos = verifier_geos.to_crs(config['verifiers']['crs_proj'])

    dist_df = selected_geos.geometry.apply(
            lambda g: verifier_geos.distance(g)
        ).reset_index().melt(
            id_vars='detection_id',
            var_name='verifier_ix',
            value_name='distance'
        )
    dist_df['distance'] = dist_df['distance'].apply(meters_to_miles)

    logging.debug(f"Head of dataframe with distances to verifiers: \n{dist_df.head()}")

    return verifier_df, dist_df

def event_selector(config, new_events, verifier_df, dist_df):
    selected_events = pd.DataFrame(columns=new_events.columns)

    if config['event_selector'].get('top_n'):
        logging.info(f"Selecting events using top {config['event_selector']['top_n']} detections method")
        new_events.sort_values('score', ascending=False, inplace=True)
        selected_events = pd.concat([selected_events, new_events[:config['event_selector']['top_n']]])

    elif config['event_selector'].get('counties'):
        logging.info(f"Selecting events using county counts method")
        for county, top_n in config['event_selector']['counties'].items():
            cty_events = new_events.loc[new_events['county_name']==county, ].copy()
            cty_events.sort_values('score', ascending=False, inplace=True)
            selected_events=pd.concat([selected_events, cty_events[:top_n]])

    elif config['event_selector'].get('closest_n_per_verifier'):
        logging.info(f"Selecting events using closest_n_per_verifier method")

        max_dist = config['event_selector']['max_dist']
       
        for id, verifier in verifier_df.iterrows():
            verifier_detections = dist_df[dist_df['verifier_ix'] == id]

            #threshold driving distance, for simplicity this just does closeness to the closest verifier
            close_detections = verifier_detections[verifier_detections['distance'] <=  max_dist]

            #check if the verifier is any of the nearest ones
            #don't add the same event twice
            filtered_df = close_detections[~(close_detections['detection_id'].isin(selected_events['detection_id']))]
            
            #prioritize score of the events that are close enough
            filtered_events = new_events[new_events['detection_id'].isin(filtered_df['detection_id'])]

            #filter events from counties we dont want
            filtered_events = filtered_events[~(filtered_events['county_name'].isin(config['event_selector']['exclude_counties']))]
            
            filtered_events.sort_values('score', ascending=False, inplace=True)
            selected_events=pd.concat([selected_events, filtered_events[:config['event_selector']['closest_n_per_verifier']]])

    else:
        return ValueError('Must specify either overall top_n, county counts or closest verifier counts in event_selector config!')

    logging.info(f"Total of {len(selected_events)} events selected to send to verifiers.")
    logging.debug(f"Head of selected events: {selected_events.head()}")
    
    return selected_events

def add_nearest_verifiers(config, selected_events, verifier_df, dist_df):
    
    num_val = config['verifiers']['num_nearest']
    result_list = []

    for id in dist_df['detection_id'].unique():
        res = {'detection_id': id}
        val_num=1
        for _, val in dist_df.loc[dist_df['detection_id']==id, ].sort_values('distance')[:num_val].iterrows():
            val_info = verifier_df.loc[val['verifier_ix']]
            dist = val['distance']
            res['verifier%s_name' % val_num] = f"{val_info['first_name']} {val_info['last_name']}"
            res['verifier%s_addr' % val_num] = f"{val_info['address_street']}, {val_info['address_city']}, {val_info['address_state']} {val_info['address_zip']}"
            res['verifier%s_distance' % val_num] = dist
            val_num += 1
        result_list.append(res)

    res_df = pd.DataFrame(result_list).set_index('detection_id')
    return selected_events.set_index('detection_id').join(res_df).reset_index()

def assign_drivers(selected_events, driver_geos):
    # probably some greedy thing like loop over drivers, take closest unassigned event, etc.
    # until each driver has right number (might be better to do at event_selector() level)
    # if we need to do this...
    raise NotImplementedError()

def save_to_drive(config, secrets, conn, drive, selected_events):
    today = datetime.now().strftime(config['time_format'])
    folder_id = gd.mkdir(drive, today, parent_folder_id=config['google_drive']['parent_id'])['id']
    os.mkdir(f'/tmp/{today}')

    export_cols = [
        'run_id', 'detection_id', 'location_id', 'image_date', 'detection_timestamp', 
        'lat_center', 'lon_center', 'est_size_acres',
        'county_name', 'city_town_name',
        'score', 'gdrive_image_url', 'Use this detection',  'google_maps_url', 'bing_maps_url',
        'verifier1_name', 'verifier1_addr', 'verifier1_distance',
        'verifier2_name', 'verifier2_addr', 'verifier2_distance',
        'verifier3_name', 'verifier3_addr', 'verifier3_distance'
    ]

    db_cols = [c for c in export_cols if c not in ['image_date','Use this detection','google_maps_url', 'bing_maps_url','location_id']]

    for ev_ix, event in selected_events.iterrows():
        # rename the image to the desired format: {county}_{town}_{detection_id}.png
        location_id = f"{event['county_name']}_{event['city_town_name']}_{event['detection_id']}"
        img_folder_id = gd.mkdir(drive, location_id, parent_folder_id=folder_id)['id']
        new_loc = f"/tmp/{today}/{location_id}.jpeg"
        os.system(f"""cp "{event['image_path']}" "{new_loc}" """)
        #write detection bounding box to images
        x = event['bbox_center_x']
        y = event['bbox_center_y']
        w = event['bbox_width']
        h = event['bbox_height']
        xl = x - w/2
        yb = y - h/2
        rec = np.array([xl, yb, w, h])*config['tilesize']
        
        fig, ax = plt.subplots(1, 3, figsize=(18,15))
        for a in ax:
            a.xaxis.set_visible(False)
            a.yaxis.set_visible(False)

        im_load = image.imread(new_loc)
        rect = patches.Rectangle((rec[0], rec[1]), rec[2], rec[3], facecolor='none', 
                                linewidth=2, alpha=0.8, edgecolor='#C41E3A')
        ax[0].add_patch(rect)
        ax[0].imshow(im_load)


        zoom = config['gmaps']['zoom_level']
        size = config['gmaps']['map_size']
        key = secrets['api_keys']['gmaps']
        lat = event['lat_center']
        lon = event['lon_center']
        
        r = requests.get(f'https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&markers={lat},{lon}&zoom={zoom}&size={size}x{size}&maptype=hybrid&key={key}')
        if r.status_code == 200:
            sat_map = Image.open(BytesIO(r.content))
            ax[1].imshow(sat_map)
            logging.trace(f"Google maps satellite image successfully retrieved for {location_id} event.")
        else:
            logging.warn(f"Google Maps satellite image API request didn't work for event: {location_id}")

        r = requests.get(f'https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&markers={lat},{lon}&zoom={zoom}&size={size}x{size}&maptype=roadmap&key={key}')
        if r.status_code == 200:
            road_map = Image.open(BytesIO(r.content))
            ax[2].imshow(road_map)
            logging.trace(f"Google maps road image successfully retrieved for {location_id} event.")
        else:
            logging.warn(f"Google Maps road image API request didn't work for event: {location_id}")

    
        fig.suptitle(f'{location_id} ( North \u2b06 )', y=0.68)
        fig.supxlabel('Note: the detection in the red box in left image may not be centered, but is the same location as the centered red marker in the middle and right images', y=0.3)
        fig.savefig(new_loc, bbox_inches='tight')

        # note: open up link sharing to allow for embedding via mail merge without google login
        img_id = gd.upload_from_file(drive, new_loc, parent_folder_id=img_folder_id, public_viewable=True)['id'] # alternateLink
        logging.trace(f"Image succesfully uploaded for devent: {location_id}")
        selected_events.loc[ev_ix, 'gdrive_image_url'] = gd.image_embed_url(img_id)
        selected_events.loc[ev_ix, 'location_id'] = location_id
        os.system(f'rm "{new_loc}"')

    selected_events[db_cols].pg_copy_to('sent_to_elpc', conn, schema=config['db_schema'], index=False, if_exists='append')
    logging.info(f"Info for {len(selected_events)} selected events that were sent to verifiers written to 'sent_to_elpc' table in database")
    
    selected_events['Use this detection'] = ''
    upload_df = selected_events[export_cols].copy()
    upload_df = round_df_cols(upload_df, config['google_drive']['round_cols'])

    # upload events to google drive as a CSV
    upload_df.to_csv(f'/tmp/detections_for_verification_{today}.csv', index=False)
    csv_url = gd.upload_from_file(drive, f'/tmp/detections_for_verification_{today}.csv', parent_folder_id=folder_id)['alternateLink']
    logging.info(f"Images for {len(selected_events)} selected events written to google drive")


    # also append them to the end of the google sheet
    gd.append_to_gsheet(
        config['google_drive']['creds_json'],
        upload_df,
        config['google_drive']['gsheet_key'],
        check_header=True
    )
    logging.info(f"Info for {len(selected_events)} selected events written to detections google sheet")

    return csv_url

def run_elpc(config, secrets, run_id):
    conn, tunnel = db_connect(secrets) # db_connect imported from add_predictions_to_sql

    try:
        logging.info("loading events")
        all_events = load_events(config, conn, db_schema=config['db_schema'],run_id=run_id, run_nickname=config['run_nickname'])
        logging.info('filtering previously seen events')
        new_events = filter_prev_seen(config, conn, config['db_schema'], all_events)
        logging.info('filtering duplicate detections')
        new_events = filter_duplicate_detections(config, new_events)

        if new_events.shape[0] == 0:
            logging.warn("No new events detected in this run.")

        else:
            # selected_events = assign_drivers(selected_events, driver_geos)
            new_events['gdrive_image_url'] = '' # placeholder column for URL after loading image to google drive
            new_events['location_id'] = '' # placeholder for county_city_detection-id identifier for elpc
            new_events['google_maps_url'] = new_events.apply(
                lambda r: gd.gmaps_url(r['lat_center'], r['lon_center']),
                axis=1)
            new_events['bing_maps_url'] = new_events.apply(
                lambda r: gd.bing_maps_url(r['lat_center'], r['lon_center']),
                axis=1)
            logging.info('Filtering for distance to verifier')
            verifier_df, dist_df = make_dist_df(config, new_events)
            selected_events = event_selector(config, new_events, verifier_df, dist_df)
            print('Done, len ', selected_events.shape[0])
            if selected_events.shape[0] == 0:
                logging.warn("No new events were selected to write to the Drive in this run. Revisit the selection criteria.")
            else:
                selected_events = add_nearest_verifiers(config ,selected_events, verifier_df, dist_df)
                logging.info('adding selected events to drive')
                drive = gd.connect(config['google_drive']['creds_json'])
                save_to_drive(config, secrets, conn, drive, selected_events)
    finally:
        conn.dispose()
        if tunnel:
            tunnel.close()