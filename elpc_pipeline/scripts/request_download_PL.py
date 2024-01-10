import os
import warnings
import shutil 
import requests
import numpy as np 
import datetime as dt
import time
import random
import pandas as pd
import sys
import argparse
import yaml

from pathlib import Path
from collections import Counter
from json.decoder import JSONDecodeError
from tqdm import tqdm 
from requests.auth import HTTPBasicAuth

from calendar import monthrange

import logging
logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 20 # max number of times to try requesting Planet API for clip before moving on
URL = 'https://api.planet.com/compute/ops/orders/v2' # URL for Planet API

def create_date(year, month, day):
    """Convert given year-month-day into ISO-8601 date representation. 
     E.g., 2015-03-04T00:00:00.000Z is a valid ISO-8601 date representation."""
    if month > 12: 
      year += 1
      month = month % 12
    date = str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2) + "T00:00:00.000Z"
    return date

def year_month(year, month):
    return str(year) + '-' + str(month).zfill(2)

def year_month_day(year, month, day):
    return str(year) + '-' + str(month).zfill(2) + '-'+ str(day).zfill(2)

def n_days(year, month):
    """Get number of days in given month of given year"""
    if month > 12: 
      year += 1
      month = month % 12
    return monthrange(year, month)[1]

def get_end_date(year, start_month, start_day, n_months, num_days):
    """
    Given a start data and desired number of months and days, returns the end date of the period.
    Should only be used to return a date range less than one year
    Returns: the ISO-8601 formatted date.
    """
    assert n_months < 12 and num_days+(n_months*30) < 365, "This shouldn't be used to pull data for multiple years"
    
    if start_month+n_months>12:
        end_year = year+1
        end_month =  (start_month+n_months) % 12
    else:
        end_year = year
        end_month = start_month+n_months
    
    if n_days(end_year, end_month) < num_days:
        logging.warn('Num days is longer than the month of start_month+n_months, this will result in the actual retrieved month being later than the specified month.')

    end_date = dt.date(end_year, end_month, start_day)+dt.timedelta(days=num_days)

    return create_date(end_date.year, end_date.month, end_date.day)

def gen_box_coords(lat, lon, height=2000, width=2000):
    """
    uses this approximate formula to convert between distances and degrees which just uses a straight line in lat lon
    adjusted for changes in latitude
    https://jonisalonen.com/2014/computing-distance-between-coordinates-can-be-simple-and-fast/
    Args:
        lat (float): latitude in decimal degrees
        lon (float): longitude in decimal degrees
        height (float): height of box in meters north/south
        width (float): width of box in meters east/west
    
    Returns:
        box polygon coordinates
    """
    deglen = 110.25*1000 # distance of one degree at equator in mwrwea

    lat_height = height/deglen
    lon_width = width/deglen
    w = lon_width / 2
    h = lat_height / 2

    # format is [[left, bbottom], [right, bottom], [right, top], [left, top], [left, bottom]]

    box_coords = [[lon - w, lat - h],
                  [lon + w, lat - h],
                  [lon + w, lat + h],
                  [lon - w, lat + h],
                  [lon - w, lat - h]]
    return box_coords


def create_jobs(df, year=2020, start_month=1, start_day=1, n_months=0, n_days = 1,
                item_type='PSScene', asset_type='ortho_visual', 
               lat_name='Latitude', lon_name='Longitude', 
              save_path='./', height=2000, width=2000):
    """
    Creates jobs to submit to dataloop. 
    Takes a series of locations, date parameters, and other planet imagery params from the config (e.g., area of interest height/width) 
    Returns, a list of jobs that the planet api will accept to search and return the specified data for
    those parameters.
    """

    # Check that data frame has an ID olumn
    if 'id' not in df.columns:
       raise IndexError("Locations dataframe should have a column called 'id', which has the desired location ID for each row.")
    
    jobs = []

    for i, location in df.iterrows():
        
        # Get location ID - this will be used to create/add to the folder for that location
        loc_dir_name = 'loc_' + str(location.id)
        directory = os.path.join(save_path, loc_dir_name)

        # Get start and end dates for imagery to look for, in the correct format
        start_date = create_date(year, start_month, start_day)
        end_date = get_end_date(year, start_month, start_day, n_months, n_days)

        # Get coordinates for box that will define the area of itnterest to look for imagery
        lat, lon = location[lat_name], location[lon_name]
        coords = gen_box_coords(lat, lon, height, width) 

        # Pull it all together in a config for the job to submit to planet
        config = {
            'start_date': start_date,
            'end_date': end_date,
            'item_type': item_type,
            'asset_type': asset_type,
            'out_dir': os.path.join(
                directory, f'{year_month_day(year, start_month, start_day)}'
            ),
            'coordinates': coords
        }

        jobs += [config]

    logging.info(f"{len(jobs)} jobs created for Planet")
    logging.debug(f"Print-out of one of the job configs: {jobs[0]}")

    return jobs

def search_api(job, planet_key, cloud_cover=.2):
    """
    Args:
        Job config to submit to planet
        cloud_cover (double, 0-1): filter for images at most this cloudy (for use with PSScene3Band imagery)

    Returns:
        A list of item IDs in Planet's system that matched the search filters (requested area/bbox, date filter, cloud_cover, etc.)
    """
    if len(job['coordinates']) > 1:
        job['coordinates'] = [job['coordinates']]

    geo_json_geometry = {
        "type": "Polygon",
        "coordinates": job['coordinates']
    }

    # filter for items the overlap with our chosen geometry
    geometry_filter = {
      "type": "GeometryFilter",
      "field_name": "geometry",
      "config": geo_json_geometry
    }

    # filter images acquired in a certain date range
    date_range_filter = {
      "type": "DateRangeFilter",
      "field_name": "acquired",
      "config": {
        "gte": job['start_date'],
        "lte": job['end_date']
      }
    }

    # filter images based on cloud tolerance
    cloud_cover_filter = {
      "type": "RangeFilter",
      "field_name": "cloud_cover",
      "config": {
        "lte": cloud_cover
      }
    }

    permission_filter = {
       "type":"PermissionFilter",
       "config": [
          "assets:download"
       ]
    }

    usable_data_filter = cloud_cover_filter

    # create a filter that combines our geo and date filters
    combined_filter = {
      "type": "AndFilter",
      #"config": [geometry_filter, date_range_filter, permission_filter]
      "config": [geometry_filter, date_range_filter, usable_data_filter, permission_filter]
    }

    # Search API request object
    search_endpoint_request = {
      "item_types": [job['item_type']],
      "asset_types": [job['asset_type']],
      "filter": combined_filter
    }

    attempts = 0

    # while loop to keep trying to search APi until attempts < MAX_ATTEMPTS
    while attempts < MAX_ATTEMPTS:
        result = \
          requests.post(
            'https://api.planet.com/data/v1/quick-search',
            auth=HTTPBasicAuth(planet_key, ''),
            json=search_endpoint_request)
        if result.status_code != 429:
            if result.status_code != 200:
                raise Exception(result)
            break

        # If rate limited, wait and try again
        time.sleep((2 ** attempts) + random.random())
        attempts = attempts + 1

    if 'json' not in result.headers.get('Content-Type'):
        raise Exception(f"{result} in search_api()")

    ids = []
    for result in result.json()['features']:
        ids.append(result['id'])

    return ids

def create_clip(job, item_ids, composite=True):
  if 'visual' in job['asset_type']:
    product_bundle = 'visual'
  else:
    product_bundle = 'analytic'
  clip = {
      "name": "clip",
      "order_type": "partial",
      "products": [
        {
          "item_ids": item_ids,
          "item_type": job['item_type'],
          "product_bundle": product_bundle
        }
      ],
      "tools": [
        {
          "clip": {
            "aoi": {
              "type": "Polygon",
              "coordinates": job['coordinates']
          }
        }
        }
      ]
    }

  if composite:
    clip['tools'].append({"composite":{}})

  return clip

def remove_existing(job, ids): 
  """
  Remove all jobs which already have corresponding files (e.g., 
  if you've run the download script multiple times on overlapping sets
  of images). 
  """
  if not os.path.exists(job['out_dir']): 
    return ids 

  to_remove = []
  for i, id in enumerate(ids): 
    if any([f.startswith(id) for f in os.listdir(job['out_dir'])]):
      to_remove.append(i)

  return list(np.delete(np.array(ids), to_remove))


def create_orders(jobs, planet_key, cloud_percent=0.2):
    """Create list of orders to send to planet. Remove 
    orders which have a corresponding image already downloaded.
    """
    order_ids = [] # list of arrays 

    start = time.time()
    for job in tqdm(jobs, desc='creating orders'): 
        job_ids = search_api(
            job,
            planet_key,
            cloud_cover=cloud_percent
        )
        job_ids = remove_existing(job, job_ids) # remove completed jobs
        order_ids.append(job_ids) 

    logging.info(f'Orders created. ({time.time() - start:0.1f}s)')
    return order_ids 


def submit_requests(jobs, order_ids, planet_key, composite=True): 
  """Submit requests to Planet."""
  responses = {}
  start = time.time()
  desc = 'submitting jobs'
  for i, (job, ids) in tqdm(enumerate(zip(jobs, order_ids)), desc=desc):

    clip = create_clip(job, ids, composite)
    response_order = requests.post(
        URL,
        auth=HTTPBasicAuth(planet_key, ''),
        json=clip
    )
    responses[i] = {
      'status': 'requested', 
      'order': response_order, 
      'out_dir': job['out_dir']
    }

  logging.info(f'Submitted {len(jobs)} requests. ({time.time() - start:0.1f}s)')
  return responses 

def submit_requests_GCS(jobs, order_ids, key, bucket, planet_key, composite=True):
  """
  Args:
      coordinates (list): results of gen_box_coords(), a list of lat/lon pairs that define a polygon to clip to
      item_ids (list): a list of Planet item ids to clip and composite together
      item_type (string): either 'PSScene3Band' or 'PSScene4Band', will determine whether 3 or 4 band imagery is returned. 4 band imagery more often respects the usable data (clear_percent) filter.

  Returns:
      A string, the UUID of the order that was created
  """

  # Creating an Order: https://developers.planet.com/docs/orders/reference/#operation/orderScene
  responses = {}
  start = time.time()
  desc = 'submitting jobs'
  for i, (job, ids) in tqdm(enumerate(zip(jobs, order_ids)), desc=desc):
    clip = create_clip(job, ids, composite)
    clip['delivery'] =  {
      "google_cloud_storage": {
          "bucket": bucket,
          "credentials": key,
          "path_prefix": job['out_dir']
      }
  }
    response_order = requests.post(
        URL,
        auth=HTTPBasicAuth(planet_key, ''),
        json=clip
    )
    responses[i] = {
      'status': 'requested', 
      'order': response_order, 
      'out_dir': job['out_dir'],
      'clip': clip
    }

  logging.info(f'Submitted {len(jobs)} requests. ({time.time() - start:0.1f}s)')
  logging.debug(f"Responses from the requests: {responses}")
  return responses

def handle_download(jobs, session, planet_key, max_wait_time=256, verbose=True):
    """In theory, call this on the list of jobs to download them all. This 
    should call the appropriate functions with the appropriate wait times, 
    backing off expoenentially if the server is receiving too many requests. 
    
    In practice, I just do all these steps manually so I can ensure that
    everything is working. Welcome to the jungle. 
    """

    # Create orders by querying db, then submit requests 
    order_ids = create_orders(jobs, planet_key)
    responses = submit_requests(jobs, order_ids, planet_key)

    # Alternate between updating requests and downloading 
    # ready resources. If queries to API are rate limited, 
    # apply exponential backoff to request times. 
    wait_time = 1
    rate_limited = False 
    while not all_jobs_downloaded(responses): 
        if rate_limited: 
            time.sleep(wait_time)
        if wait_time < max_wait_time:
            wait_time *= 2
    responses, rl1 = check_requested(responses)
    responses, rl2 = check_accepted(responses, session)
    responses, rl3 = download_successes(responses, session)
    rate_limited = rl1 or rl2 or rl3
    if verbose: 
        print(status_counter(responses))

    return responses

def status_counter(responses): 
  """Print the status of each response"""

  stati = [] # that a word?
  for v in responses.values():
    stati.append(v['status'])

  return Counter(stati)

def check_requested(responses):
  """Check responses which have been requested but not yet accepted. 
  Update 
  """
  
  rate_limited = False
  for k, v in responses.items():

    if v['status'] != 'requested':
      continue 

    if v['order'].status_code == 429: 
      rate_limited = True 
      v['status'] = 'timeout' 
    elif v['order'].ok: 
      v['status'] = 'accepted'
      v['id'] = v['order'].json()['id']
      # print(f'Order {v["id"]} accepted.')
    else: 
      v['status'] = 'failed'
      logging.warning(f'Failed with code {v["order"].status_code}. \n{v["order"].content}')

  return responses, rate_limited

def check_all_running(responses, session): 
  """Returns true iff all responses are in 'running' state"""

  for v in responses.values():
    r = session.get(
        os.path.join('https://api.planet.com/compute/ops/orders/v2', f'{v["id"]}')
    )
    if r.json()['state'] != 'running':
      return False 

  return True 


def extract_json_results(json_content):
  results = []
  for result in json_content:
    if result['name'].endswith('.json'):
      results.append(result)
    if result['name'].endswith('AnalyticMS_clip.tif'):
      results.append(result)

  return results 



def check_accepted(responses, session):
  """Check all responses with `accepted' status. Update to success 
  if warranted"""

  rate_limited = False 
  for v in responses.values():
    if v['status'] != 'accepted': 
        continue 

    r = session.get(
        os.path.join('https://api.planet.com/compute/ops/orders/v2', f'{v["id"]}')
    )
    try: 
        if r.status_code == 429:
            rate_limited = True
            break 
        elif r.json()['state'] in ['success', 'partial']: 
            results = extract_json_results(r.json()['_links']['results'])
            v['media'] = {}
            for result in results: 
                v['media'][result['name']] = {'result': result, 'response': None}
            v['status'] = 'success'
        elif r.json()['state'] == 'failed':
            v['status'] = 'failed'
    except JSONDecodeError:
        rate_limited = True 

  return responses, rate_limited

def all_jobs_finished(responses):
  """return true iff all jobs are finished by Planet"""
  for v in responses.values():
    if not v['status'] == 'success' or v['status'] == 'failed':
      return False 
  return True

def all_jobs_downloaded(responses): 
  """return true iff all jobs are downloaded"""

  for v in responses.values():
    if not v['status'] == 'downloaded':
      return False 
  return True


def all_results_downloaded(results, dir):
  """returns true iff all the results are downloaded for a 
  given request"""

  for filename in results.keys():
    if not os.path.exists(
        os.path.join(dir, filename.split(os.sep)[-1])
    ): 
      return False
  return True
  

def download_successes(responses, session):
  """Download the responses with status `success'
  """
  rate_limited = False
  for k, v in responses.items():

    if v['status'] != 'success': 
      continue 

    for name, info_dict in v['media'].items(): 

      if os.path.exists(
          os.path.join(v['out_dir'], name.split(os.sep)[-1])
      ): 
        continue 

      if info_dict['response'] is None: 
        # submit request 
        token = info_dict['result']['location'].partition('?token=')[2]
        params = (
          ('token', token),
        )
        download_response = session.get(
          'https://api.planet.com/compute/ops/download/', 
          params=params, stream=True
        )
        info_dict['response'] = download_response
#        v['media'][name]['response'] = download_response # set info_dict['response']

      if info_dict['response'].status_code == 200:
          if not os.path.exists(v['out_dir']): 
            os.makedirs(v['out_dir'])
          with open(os.path.join(v['out_dir'], name.split(os.sep)[-1]), 'wb') as f:
              download_response.raw.decode_content = True
              shutil.copyfileobj(download_response.raw, f)

      elif info_dict['response'].status_code == 429: 
        rate_limited = True  

    if all_results_downloaded(v['media'], v['out_dir']):
      v['status'] = 'downloaded'

  return responses, rate_limited


def planet_api_pipeline(config_dict, secrets_dict):
  locations_df = pd.read_csv(config_dict['locations_filename'])
  logger.info("Read in CAFO locations .csv")
  logger.debug(locations_df.head())

  jobs = create_jobs(
    locations_df, 
    year=config_dict['year'], 
    start_month=config_dict['month'], 
    start_day=config_dict['day'], 
    n_months=config_dict['n_months'], 
    n_days=config_dict['n_days'],
    save_path=config_dict['gcs_save_path'],
    item_type=config_dict['item_type'], 
    asset_type=config_dict['asset_type'], 
    lat_name=config_dict['lat_name'], 
    lon_name=config_dict['lon_name'], 
    height=config_dict['aoi_height'], 
    width=config_dict['aoi_width']
  )
  gcs_key = secrets_dict['api_keys']['gcs']
  planet_key = secrets_dict['api_keys']['planet']

  order_ids = create_orders(jobs, planet_key, cloud_percent=config_dict['cloud_percent'])
  responses = submit_requests_GCS(jobs, order_ids, gcs_key, config_dict['gcs_bucket'], planet_key)
  # Create session
  retry = 0
  logging.info('waiting for jobs to be finished')
  finished = False
  while not finished:

    responses, ratelimited = check_requested(responses)
    wait = 0
    for v in responses.values():
      if v['status'] == 'accepted': 
        wait +=1
    #we wait at least 5 minutes before checking to see if things are finished
    waittime = (60*5 + min(60*retry, 300))
    logging.info('pausing for ', waittime, ' seconds')
    #wait for a set time for each job plus some amount of time to stop timeouts when the number'
    # of jobs goes down 
    time.sleep(waittime)

    if ratelimited:
      for k in responses:
        r = responses[k]
        if r['status'] == 'timeout':
          response_order = requests.post(
              URL,
              auth=HTTPBasicAuth(planet_key, ''),
              json=r['clip'])
          r['status'] =  'requested'
          r['order'] = response_order
          responses[k] = r
    
    #remake session each retry cycle to make sure we dont timeout
    session = requests.Session()
    session.auth = (planet_key, '')
    responses, rl = check_accepted(responses, session)
    logging.info(status_counter(responses))
    retry += 1
    finished = all_jobs_finished(responses)
  # responses, rl = download_successes(responses, session)
  # print(status_counter(responses))
