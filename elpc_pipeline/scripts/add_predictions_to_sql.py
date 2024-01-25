import os, sys
import datetime as dt
import argparse
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import yaml
import ohio.ext.pandas
import sshtunnel
import geopandas as gpd
import warnings
from tqdm import tqdm
import uuid

from shapely.geometry import Point
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

"""
database columns:
    run_id INT,
    run_nickname STR,
    image_date TIMESTAMP, -- timestamp when the image was actually captured
    detection_timestamp TIMESTAMP, -- timestamp when model was used to make inference
    image_path VARCHAR, -- path to the image on disk (though will only be relevant for current run?)
    image_location INT, -- that is, the location identifier to use across runs
    lat_center FLOAT, -- center of the detected application in lat/lon
    lon_center FLOAT,
    lat_max FLOAT,
    lat_min FLOAT,
    lon_max FLOAT,
    lon_min FLOAT,
    est_size_acres FLOAT,
    -- bbox cols in image coords (e.g., 0-1)
    bbox_center_x FLOAT,
    bbox_center_y FLOAT,
    bbox_width FLOAT,
    bbox_height FLOAT,
    county_name VARCHAR,
    city_town_name VARCHAR,
    score FLOAT

"""

def db_connect(secrets):
    db_params = secrets['db']

    if db_params.get('use_tunnel'):
        tunnel = sshtunnel.SSHTunnelForwarder(db_params['ssh_host'],
                    ssh_username=db_params['ssh_user'],
                    ssh_password=db_params['ssh_pass'],
                    remote_bind_address = (db_params['host'], db_params['port']),
                    local_bind_address=('localhost', db_params['local_port']),
                    ssh_port = db_params['ssh_port']
                )
        tunnel.start()
        eng = create_engine("postgresql+psycopg2://{sql_user}:{sql_pass}@0.0.0.0:{local_port}/{dbname}?sslmode=allow".format(
                sql_user=db_params['user'],
                sql_pass=db_params['pass'],
                local_port=db_params['local_port'],
                dbname=db_params['dbname']
            ))
        # Testing out db connection
        try:
            with eng.connect() as conn:
                result = conn.execute(text("select * from information_schema.tables limit 10;"))
                if result.rowcount > 0:
                    logging.debug("Database connection successful.")
                else:
                    logging.error("Database connected but no data retrieved. Check the database structure and connection")
        except SQLAlchemyError as err:
            logging.error("Error in connecting to database, because of the following cause: ", err.__cause__)
    else:
        tunnel = None
        eng = create_engine('postgresql://{user}:{password}@{host}:{port}/{dbname}'.format(
                host=db_params['host'],
                port=db_params['port'],
                dbname=db_params['dbname'],
                user=db_params['user'],
                password=db_params['pass']
            ))
        try:
            with eng.connect() as conn:
                result = conn.execute(text("select * from information_schema.tables limit 10;"))
                if result.rowcount > 0:
                    logging.debug("Database connection successful.")
                else:
                    logging.error("Database connected but no data retrieved. Check the database structure and connection")
        except SQLAlchemyError as err:
            logging.error("Error in connecting to database, because of the following cause: ", err.__cause__)
    
    return eng, tunnel


def main(config, secrets, run_id):
    year_month_day = str(config['year']) + '-' + str(config['month']).zfill(2) + '-'+ str(config['day']).zfill(2)
    loc_names = [f for f in os.listdir(os.path.join(config['root'], config['gcs_save_path'])) if 'loc_' in f]
    cities_gdf = gpd.read_file(config['cities_towns_shapefile'])
    cities_gdf = cities_gdf.to_crs('EPSG:4326')
    logging.info(f"Read cities/towns shapefile, with {len(cities_gdf)} rows")
    logging.debug(f"Head of cities DF: \n{cities_gdf.head()}")
    
    fields_gdf = gpd.read_file(config['wdnr_fields_shapefile'])
    fields_gdf = fields_gdf.to_crs('EPSG:4326')
    logging.info(f"Read fields shapefile, with {len(fields_gdf)} rows")
    logging.debug(f"Head of fields DF: \n{cities_gdf.head()}")

    detections = []
    for loc in tqdm(loc_names):
        results_path = os.path.join(config['root'], config['gcs_save_path'], loc, year_month_day, 'labels')    
        images_path = os.path.join(config['root'], config['gcs_save_path'], loc, year_month_day, 'images')
        if not os.path.exists(images_path) or len([x for x in os.listdir(images_path) if '.jpeg' in x]) == 0:
            logging.warn(f"No images with detections exist for location ID {loc}")
            continue
        image_coordinates = pd.read_csv(os.path.join(images_path, 'coordinates.csv'))
        labels_files = [txt for txt in os.listdir(os.path.join(results_path, f"exp{run_id}", 'labels')) if '.txt' in txt]
        for txt in labels_files: 
            no_ext = os.path.splitext(txt)[0]
            loc = no_ext.split('_')[1] 
            date = dt.datetime.strptime(no_ext.split('_')[2], '%Y-%m-%d')
            tile_id = [no_ext.split('_')[3], no_ext.split('_')[4]] #location of tile in pixels or meters
            image_name =  no_ext+'.jpeg'
            image_full_path = os.path.join(images_path, image_name)
            coords = image_coordinates[image_coordinates['jpeg_name'] == image_name].iloc[0]
            with open(os.path.join(results_path, f"exp{run_id}", 'labels', txt), 'r') as f: 
                for l in [l.strip() for l in f.readlines()]: 
                    arr = l.split(' ')
                    plbl = [float(x) for x in arr[1:-1]] #x_center,y_center,w,h
                    conf = float(arr[-1]) 

                    lat_center = coords['lat_max'] - (coords['lat_max'] - coords['lat_min'])*plbl[1] 
                    lon_center = coords['lon_min'] + (coords['lon_max'] - coords['lon_min'])*plbl[0]
                    lat_height = (coords['lat_max'] - coords['lat_min'])*plbl[3]
                    lon_width = (coords['lon_max'] - coords['lon_min'])*plbl[2]

                    lat_max = lat_center + lat_height/2
                    lat_min = lat_center - lat_height/2
                    lon_max = lon_center + lon_width/2
                    lon_min = lon_center - lon_width/2

                    deglen = 110.25 # distance of one degree at equator in km
                    height = lat_height*deglen
                    width = lon_width*deglen
                    area_km2 = height*width
                    area_acre = area_km2*247.105381 #acres/sq km 
                    city_df = cities_gdf.sjoin(gpd.GeoDataFrame([{'geometry': Point(lon_center, lat_center)}], crs='epsg:4326'),how='inner')
                    if city_df.shape[0] == 0:
                        city = None,
                        county= None
                    else:
                        city = city_df['MCD_NAME'].iloc[0]
                        county = city_df['CNTY_NAME'].iloc[0]
                    
                    field_df = fields_gdf.sjoin(gpd.GeoDataFrame([{'geometry': Point(lon_center, lat_center)}], crs='epsg:4326'),how='inner')
                    
                    if field_df.shape[0] == 0:
                        farm_name = None,
                        field_name = None,
                    else:
                        if field_df.shape[0] > 1:
                            logging.warn('Detection matches more than one field')
                            field_df = field_df.sample(frac=1.0)
                        farm_name = field_df['FARM_NAME'].iloc[0]
                        field_name = field_df['FIELD_NAME'].iloc[0]

                    detections.append(pd.DataFrame([{'run_id': run_id,
                                    'run_nickname': config['run_nickname'],
                                    'image_path': image_full_path, 
                                    'image_date': date, 
                                    'detection_timestamp':dt.datetime.now(),
                                    'image_location': int(loc+tile_id[0]+tile_id[1]), #should be unique for each image?
                                    'lat_center': lat_center,
                                    'lon_center': lon_center,
                                    'lat_max': lat_max,
                                    'lat_min': lat_min,
                                    'lon_max': lon_max,
                                    'lon_min': lon_min,
                                    'est_size_acres': area_acre, 
                                    'bbox_center_x': plbl[0],
                                    'bbox_center_y': plbl[1],
                                    'bbox_width': plbl[2],
                                    'bbox_height': plbl[3],
                                    'county_name': county,
                                    'city_town_name': city,
                                    'score': conf,
                                    'farm_name': farm_name,
                                    'field_name': field_name
                                    }]))

    
    df = pd.concat(detections, ignore_index=True)
    logging.debug(f"Head of events df to write to database: \n{df.head()}")

    try:
        conn, tunnel = db_connect(secrets)

        df.pg_copy_to('detections',
                        conn, 
                        schema=config['db_schema'], 
                        index=False, 
                        if_exists='append')
        logging.info(f"Wrote info for {len(df)} detections to the database.")
    finally:
        conn.dispose()
        if tunnel:
            tunnel.close()