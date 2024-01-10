'''
run_pipeline.py
Main pipeline runner file
'''
import yaml
import logging
import argparse
import sys
from datetime import datetime, date
import os
import json
from google.cloud import storage
import hashlib


# Importing other scripts as modules
import scripts.request_download_PL as planet
import scripts.tile_and_filter_tiffs as preprocess
import scripts.run_yolov5 as detection
import scripts.add_predictions_to_sql as writesql
import scripts.post_pred_pipeline as postpred

def load_config(config_filename):
    '''
    Reads + processes config file, returning a config dict to be passed to all parts
    of the pipeline.
    '''
    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)
        logging.info(f"Config file {config_filename} loaded.")
        logging.debug(f"Printout of loaded config:\n {config}")

    # Set dates if 'use_yesterday' or 'use_today' flags are set to true
    if config['use_yesterday']:
        yesterday = date.today() - date.timedelta(days = 1)
        config['year'] = yesterday.year
        config['month'] = yesterday.month
        config['day'] =  yesterday.day
        config['n_days'] = 1
        config['n_months'] = 0

    elif config['use_today']:
        today = date.today()
        config['year'] = today.year
        config['month'] = today.month
        config['day'] =  today.day
        config['n_days'] = 1
        config['n_months'] = 0

    if config['month'] is None or config['day'] is None:
        raise ValueError('Must specify a date or use yesterday')    
    
    return config

def load_secrets(secrets_filename):
    '''
    Reads consolidated secrets.yaml file containing all credentials/keys, outputs a
    dict to be passed to appropriate parts of the pipeline. 
    
    The expected format for secrets.yaml is shown in example_secrets.yaml
    '''

    with open(secrets_filename, 'r') as f:
        secrets = yaml.safe_load(f)

    logging.info(f"Secrets file loaded.")
    logging.debug(f"Printout of loaded secrets:\n {secrets}")

    return secrets

def add_run_to_sql(secrets, config):
    '''
    Writes run info to the pipeline_runs table in SQL.
    run_id is an auto-increment ID generated in SQL.
    Function also returns the current run_id, which is passed to other SQL-writing functions later in the pipeline
    '''
    eng, tunnel = writesql.db_connect(secrets) # Note: this db_connect function has logging statements and checks that the connection is working

    # Hash the config (to be able to know if the same config was used in the run)
    config_json = json.dumps(config)
    config_hash = hashlib.sha256(config_json.encode())
    config_hash = config_hash.hexdigest()
    logging.debug(f"Config file hashed as {config_hash}.\nData type of hash = {config_hash.type}")

    # Get the relevant date/day info for the run, to also store this separately in columns for easy retrieval
    date_val = date(config['year'], config['month'], config['day'])
    days_range_val = config['n_days']
    month_range_val = config['n_months']

    with eng.connect() as conn:
    
        conn.execute(f'''INSERT INTO {config['db_schema']}.pipeline_runs (run_nickname, config_hash, imagery_date, imagery_days_range, imagery_months_range, config)
                    VALUES ('{config['run_nickname']}', '{config_hash}' ,'{date_val}'::DATE, {days_range_val}::INT, {month_range_val}::INT, %(config)s::jsonb)
                ''', {'config': config_json})
        
        # Retrieve the run_id auto-generated for this entry
        result = conn.execute('SELECT lastval()')
        run_id = result.fetchone()[0]
        
        logging.info(f"Run written to pipeline_runs table in DB, with ID: {run_id}")

    # Close connection
    eng.dispose()
    if tunnel:
        tunnel.close()

    return run_id

def update_run_status(config, secrets, run_id, status):
    '''
    Helper func that updates the status of a run with desired status message
    '''
    eng, tunnel = writesql.db_connect(secrets)

    with eng.connect() as conn:
        conn.execute(f"UPDATE {config['db_schema']}.pipeline_runs SET run_status = '{status}' WHERE run_id = {run_id}")
        logging.info(f"Updated run_status to '{status}' for run_id {run_id}")
   
    eng.dispose()
    if tunnel:
        tunnel.close()


def main(log_filename, log_level, config_filename, secrets_filename):

    ###### SETTING UP LOGGING #####

    # Instantiate root logger (note - all logging in imported modules will inherit properties of this root logger
    # including logging level, format, and filename)
    logging.basicConfig(level=log_level) 

    # Set formatting for logs
    frmt = logging.Formatter('%(name)-30s  %(asctime)s %(levelname)10s %(process)6d  %(filename)-24s  %(lineno)4d: %(message)s', '%d/%m/%Y %I:%M:%S %p')

    # Set log file output to write to (filehandler object)
    filehdlr = logging.FileHandler(log_filename)
    filehdlr.setFormatter(frmt)
    logging.root.addHandler(filehdlr)

    # Also set stdout (terminal) as an output
    consolehdlr = logging.StreamHandler(sys.stdout)
    consolehdlr.setFormatter(frmt)
    logging.root.addHandler(consolehdlr)

    ###### LOAD/INSTANTIATE NECESSARY FILES/VARS/CONNECTIONS ######

    # Process/load config and secrets files
    config = load_config(config_filename)

    secrets = load_secrets(secrets_filename)

    # Write run to runs table in DB, get run_id:
    run_id = add_run_to_sql(secrets, config)

    # Now try the rest of the code. If it succeeds, then update run_id with success, else failure

    try:
        # Set up Google Storage client and get bucket
        client = storage.Client()
        bucket = client.get_bucket(config['gcs_bucket'])    

        # Debug connection by checking some objects in the bucket and seeing if it retrieved okay
        blobs = bucket.list_blobs()
        logging.debug(f"Objects in GCS bucket retrieved by connection: {blobs}")

        logging.info("Connected to GCS bucket")
                    
        # Get year-month-day as a var (used as input to some functions)
        year_month_day = str(config['year']) + '-' + str(config['month']).zfill(2) + '-'+ str(config['day']).zfill(2)

        ###### CHECK IF YOLOv5 code exists #####
        if not os.path.isdir('yolov5'):
            logging.error('''Yolov5 package not found in expected location. 
                            Please follow instructions in README to clone Yolov5 repo in the same directory/level as this run_pipeline.py file.''')
        else:
            logging.info('Yolov5 package found in expected place.')

        ##### RUNNING COMPONENTS OF PIPELINE #####
        planet.planet_api_pipeline(config, secrets) # Planet imagery downloader
        preprocess.main(bucket, config, year_month_day) # Image pre-processor to create tiles for model
        detection.main(config, run_id) # Running model on tiled images
        writesql.main(config, secrets, run_id) # Writing detections to SQL

        # check if different run_id specified for use in postpred
        if config['postpred_run_id'] is not None:
            postpred.run_elpc(config, secrets, config['postpred_run_id']) # Running event selection/post-prediction, finally writing events to Drive + DB
        else:
            postpred.run_elpc(config, secrets, run_id)

        # If pipeline all ran successfully, update run_status to success
        update_run_status(config, secrets, run_id, 'success')
    
    except Exception as e:
        logging.error(f"Following exception occured: {str(e)}")
        update_run_status(config, secrets, run_id, 'failure')

if __name__ == '__main__':

    # Take in config filepath, secrets filepath, and logging level as command line arguments
    parser = argparse.ArgumentParser()
    try:
        parser.add_argument('--config_file', type=str,
                        help='Path to config .yaml file for this run')
        parser.add_argument('--secrets_file', type=str,
                            default='secrets.yaml', help='Path to secrets.yaml file')
        parser.add_argument('--log_level', type=int, choices=[10, 20, 30, 40, 50],
                            default=20, help='''See Python logging levels here for more info
                            on options: https://docs.python.org/3/library/logging.html#levels.                  
                            Default = 20, or 'info' level. Debug level = 10. Use the numeric
                            value''')
    except AttributeError as e:
        sys.exit(e)
  
    args = parser.parse_args()

    # Prep for logging filename/directory
    now = datetime.now() # Current timestamp to place into the log filename
    if not os.path.exists('logs'):
        os.mkdir('logs')

    # Run main pipeline runner function
    main(log_filename=f'logs/run_{now}.log', 
         log_level=args.log_level, # See here for description of different logging levels: https://docs.python.org/3/library/logging.html#levels
         config_filename = args.config_file,
         secrets_filename = args.secrets_file)
    



