# Land Application Detection

Code for the Paper [Detecting Environmental Violations with Satellite Imagery in
Near Real Time: Land Application under the Clean Water Act](https://arxiv.org/pdf/2208.08919.pdf), in Conference of Information and Knowledge Management (CIKM), 2022. 


Our experiments and analysis were run in Google Colab. We provide the relevant code here as jupyter notebooks. 

The imagery, labels and model weights [can be found here](https://drive.google.com/drive/u/1/folders/0AJqxmVfK4hN5Uk9PVA). The `training_data` folder also contains names of the images in the test, train, and validation sets. It also contains the yaml file which should be fed to yolov5 so that it trains on the correct data. As detailed in the paper, the train/val data and test data come from different years, representing how this model is to be deployed in the field.

## Contents
  
- `planet_images_downloader.ipynb`: Jupyter notebook to download and view imagery from the Planet API.  Must specify .csv of lat/lon pairs.
- `yolov5_training_and_inference.ipynb`: We use [YOLOv5 from ultralytics](https://github.com/ultralytics/yolov5). This notebook downloads the latest version and runs training and inference. 
- `yolov5_stats.ipynb`: Calculate performance statistics (e.g., PR, AUC) for both image classification and event detection. 
- `event_detection.ipynb`: Code to aggregate image-level bounding box predictions into "event" detections, which can span multiple days. 


