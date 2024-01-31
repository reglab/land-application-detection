# Winter Spreading Detection Pipeline
This folder has the code for a pipeline that detects likely winter manure spreading events using satellite imagery and a trained/tuned [Yolov5](https://github.com/ultralytics/yolov5) object detection model. 

## Running code using the pipeline

### One-time setup
1. Clone this repository on the server/computer where you want to be running code. The instructions below assume a Linux/Ubuntu server.
2. Set up a Python virtual environment: ```python -m venv venv```
3. Activate the environment: ```source venv/bin/activate```
- ```pip install -r requirements.txt```

pre-requisite: 
- 'gcloud auth login'
- GDAL installation
- Clone yolov5 repository and install the sub-requirements: 
```
git clone -b v7.0 https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
cd ..

```

For planet imagery, need to prepare gcs key [in this way add](https://developers.planet.com/apis/orders/delivery/#delivery-to-cloud-storage) to secrets


#### Regular running
- Run command: ```python3 run_pipeline.py --config_file configs/pipeline_config.yaml --secrets_file secrets.yaml --log_level 10 ```



### Code structure/organization 

Add image of pipeline runner.

### Important things to note aboout the pipeline
- Location ID comes from the original files. So depending on the index, location ID will be set accordingly. For tracking, would recommend keeping that the same.

### Debugging and other usage tips
- Image outputs are location-
- Logging + run_ids in database. 


## Background about the model and training


Something about sql


## Add section on debugging tips

- Activating environment: ```source */bin/activate```
- pip list to check packages installed, once in the environment

This part of the code will request, tile, filter, run inference on, add detections to lc-r-2 database then create the necessary output detection files or google drive locations

To run whole pipeline update run_pipeline.sh to have correct config files and then do source run_pipeline.sh

This will run files to:
1. update config to have correct date and run_id
2. request imagery from Planet
3. filter images and tile to correct size
4. run yolov5 on tiled tifs
5. add detections to sql along with locations and other metadata
6. pull detections and add to google drive and specified local folder (for elpc and wdnr respectively)

One set of very important variables in the config are
use_yesterday and use_today. Only one can be True but both may be False. If you use one, it will automatically update the date to the current day or the previous day.