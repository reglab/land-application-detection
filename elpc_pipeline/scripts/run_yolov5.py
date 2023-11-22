import sys, os
import argparse

import datetime as dt

from tqdm import tqdm
from pathlib import Path
from yolov5.detect import run

import logging
logger = logging.getLogger(__name__)

def main(config, run_id):
    year_month_day = str(config['year']) + '-' + str(config['month']).zfill(2) + '-'+ str(config['day']).zfill(2)
    loc_names = [f for f in os.listdir(os.path.join(config['root'], config['gcs_save_path'])) if 'loc_' in f]
    for loc in tqdm(loc_names,desc='running inference on all locations'):
        images_path = os.path.join(config['root'], config['gcs_save_path'], loc, year_month_day, 'images')
        if not os.path.exists(images_path) or len([x for x in os.listdir(images_path) if '.jpeg' in x]) == 0:
            continue
        out_path = os.path.join(config['root'], config['gcs_save_path'], loc, year_month_day, 'labels')
        run(
            source=images_path,  # file/dir/URL/glob/screen/0(webcam)
            project=out_path,  # save results to project/name
            weights=config['weights'],  # model path or triton URL
            data=config['class_config'],  # dataset.yaml path
            imgsz=(config['img_size'], config['img_size']),  # inference size (height, width)
            conf_thres=config['conf_thresh'],  # confidence threshold
            iou_thres=config['iou_thresh'],  # NMS IOU threshold
            max_det=config['max_detections'],  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            save_txt=True,  # save results to *.txt
            save_conf=True,  # save confidences in --save-txt labels
            name=f"exp{run_id}"  # save results to project/name
        )

