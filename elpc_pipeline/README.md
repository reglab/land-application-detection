Detection Pipeline readme

Notes:
- ```pip install -r requirements.txt```
- Run command: ```python3 run_pipeline.py --config_file configs/pipeline_config.yaml --secrets_file secrets.yaml --log_level 10 ```

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

The shell script will also activate the correct enviroments along the way.

One set of very important variables in the config are
use_yesterday and use_today. Only one can be True but both may be False. If you use one, it will automatically update the date to the current day or the previous day.