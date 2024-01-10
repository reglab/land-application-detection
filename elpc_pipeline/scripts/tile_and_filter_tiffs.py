import os
import shutil
import time
import numpy as np
import datetime as dt

import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt
from osgeo import ogr, osr
from osgeo import gdal
from tqdm import tqdm
from google.cloud import storage
from google.cloud.storage import Blob
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

def add_sep(path_name):
    """
    Add final os.sep (usually a "/") to the given path_name
    if it doesn't yet exist.
    """
    if path_name[-1] != os.sep:
        path_name += os.sep

    return path_name

def count_nonblack_np(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.
    """
    return img.any(axis=-1).sum()


def upload_dir_to_gcloud(bucket, src_dir, dest_dir, info):
    
    for a_file in tqdm(os.listdir(src_dir), desc="uploading {} to gcloud".format(info)):
        blob_name = os.path.join(dest_dir, a_file)
        blob = Blob(blob_name, bucket)
        blob.upload_from_filename(os.path.join(src_dir, a_file))

#-------------------------------------------------------------------------------------------------------------------------------
# various methods to check if any of the steps have been previously completed
#-------------------------------------------------------------------------------------------------------------------------------
def check_previous_tiled_gcloud(bucket, gc_tiled_tif):
    """
    See if the tif files have already been broken down into tiles and stored in the gcloud bucket
    """
    files_gen = bucket.list_blobs(prefix=gc_tiled_tif)
    split_names = [file.name.split('/') for file in files_gen]
    return len([split_name for split_name in split_names 
                if 'tiled_tif' in split_name 
                and 'completed.txt' in split_name]) > 0


def check_previous_images_gcloud(bucket, gc_images):
    files_gen = bucket.list_blobs(prefix=gc_images)
    split_names = [file.name.split('/') for file in files_gen]
    return len([split_name for split_name in split_names 
                if 'images' in split_name 
                and 'completed.txt' in split_name]) > 0


def check_previous_tiled_local(local_tiled_tif):
    """
    See if the tif files have already been broken down into tiles and stored in the gcloud bucket
    """
    return os.path.isdir(local_tiled_tif) and "completed.txt" in list(os.listdir(local_tiled_tif))


def check_previous_images_local(local_images):
    """
    See if the tiled tif files have already been converted into jpeg
    """
    return os.path.isdir(local_images) and "completed.txt" in list(os.listdir(local_images))

def download_initial_tiffs(bucket, gc_initial_tiff, local_folder):
    
    # get list of files to download
    files_list = bucket.list_blobs(prefix=gc_initial_tiff)
    files_list = [a_file for a_file in files_list 
                  if "composite.tif" in a_file.name
                  and 'completed.txt' not in a_file.name]
    
    # probably a better way to do this than having almost an exact duplicate, but eff it I'm tired
    
    # create the directory if it doesn't exist
    if not os.path.isdir(local_folder):
        print("initial tiffs do not exist locally, downloading entire directory")
        os.makedirs(local_folder)
        
        # download the initial tiffs
        for a_file in tqdm(files_list, desc="downloading initial tiff"):
            file_name = a_file.name.split('/')[-1]
            file_path = os.path.join(local_folder, file_name)
            a_file.download_to_filename(file_path)
            
    # directory exists but is empty
    elif len(os.listdir(local_folder)) == 0:
        print("initial tiffs do not exist locally, downloading entire directory")
        
        # download the initial tiffs
        for a_file in tqdm(files_list, desc="downloading initial tiff"):
            file_name = a_file.name.split('/')[-1]
            file_path = os.path.join(local_folder, file_name)
            a_file.download_to_filename(file_path)
    

    # if the local directory already exists, check to see whether or not the same number of files 
    # exist in the directory as in the cloud
    else:
        print("at least some initial tiffs exist locally")
        local_files = [a_file for a_file in os.listdir(local_folder) if a_file != "completed.txt"]
        
        # find files in the cloud whose names don't appear locally
        to_download = [a_file for a_file in files_list 
                       if a_file.name.split("/")[-1] not in local_files]
        
        # find files in the cloud who have corresponding local files, but the two files are different sizes
        different_sizes = [[(a_blob, local_file) for a_blob in files_list if a_blob.name.split("/") == local_file]
                           for local_file in local_files]
        different_sizes = [a_list for a_list in different_sizes if len(a_list) > 0]
        different_sizes = [a_list[0] for a_list in different_sizes]
        different_sizes = [a_blob for a_blob, local_file in different_sizes 
                           if a_blob.size != os.path.getsize(os.path.join(local_folder, local_file))]
        to_download += different_sizes
        
        print("downloading remaining {} files".format(len(to_download)))
        
        # download
        for a_file in tqdm(to_download, desc="downloading initial tiff"):
            file_name = a_file.name.split('/')[-1]
            file_path = os.path.join(local_folder, file_name)
            a_file.download_to_filename(file_path)
        
def download_tiled_tifs(bucket, gc_tiled_tif, local_folder):
    """
    Download all files in tiled_tif in a country from GCloud storage bucket
    """
    
    # create the directory if it doesn't exist
    if not os.path.isdir(local_folder):
        os.makedirs(local_folder)
    
    else:
        shutil.rmtree(local_folder)
        os.makedirs(local_folder)
    
    # get list of files to download
    files_list = bucket.list_blobs(prefix=gc_tiled_tif)
    files_list = [a_file for a_file in files_list 
                  if "tiled_tif" in a_file.name
                  and 'completed.txt' not in a_file.name]
    
    for a_file in tqdm(files_list, desc="downloading tiled tif"):
        file_name = a_file.name.split('/')[-1]
        file_path = os.path.join(local_folder, file_name)
        a_file.download_to_filename(file_path)
            
def get_tif_coordinates(path):
    """
    Gets the bouding points of the tif tiles in the given path, then transforms
    them from their source projection to unprojected lat lon.

    Returns a dataframe with the tif names and their bounding coordinates
    """

    targetRef = osr.SpatialReference()
    targetRef.ImportFromEPSG(4326)

    tiled_tifs = os.listdir(path)
    tiled_tifs = [tif for tif in tiled_tifs if '.tif' in tif]
    outputs = []
    for tif in tiled_tifs:
        dset = gdal.Open(os.path.join(path, tif))
        raster_srs = dset.GetSpatialRef()
        coordTrans = osr.CoordinateTransformation(raster_srs,targetRef)

        width = dset.RasterXSize
        height = dset.RasterYSize
        geotransform = dset.GetGeoTransform()
        originX = geotransform[0] #left longitude
        originY = geotransform[3] #top latitude
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5] #negative value!
        maxX = originX + width*pixelWidth
        minY = originY + height*pixelHeight

        #output crs is in lat,lon. input is int lon, lat
        (ullat, ullon, ulz) = coordTrans.TransformPoint(originX, originY) 
        (lrlat, lrlon, lrz) = coordTrans.TransformPoint(maxX, minY)

        outputs.append(pd.DataFrame([{
            'name': tif,
            'jpeg_name': os.path.splitext(tif)[0]+'.jpeg',
            'lat_min': lrlat,
            'lon_min': ullon,
            'lat_max': ullat,
            'lon_max': lrlon
            }]))
    if len(outputs) > 0:
        output_df = pd.concat(outputs, ignore_index=True)
        return output_df

    else:
        return pd.DataFrame()

def split_tiff(opt, local_dir_paths, file_name, location_id, file_ind):
    """
    Break down initial large tiff files into smaller tif tiles
    """
            
    dset = gdal.Open(os.path.join(local_dir_paths['initial_tiff'], file_name))
    width = dset.RasterXSize
    height = dset.RasterYSize
    
    ###TODO: fix this issue: Planet sometimes sends us absurdly large tiles that aren't clipped, skipping for now
    if width > 10*opt['tilesize'] or height > 10*opt['tilesize']:
        print('Warning: this image has more than 10 tiles per side, skipping')
        return 

    for i in tqdm(range(0, width, opt['tilesize']), desc='Splitting'):
        for j in range(0, height, opt['tilesize']):
            w = min(i + opt['tilesize'], width) - i
            h = min(j + opt['tilesize'], height) - j

            #if the tilesize is too small, skip this tile
            if w < opt['min_tilesize'] or h < opt['min_tilesize']:
                continue
            
            output_file = '{}_{}_{}_{}.tif'.format(location_id, file_ind, i, j)
            # Create smaller tif file of subregion
            gdal.Translate(os.path.join(local_dir_paths['tiled_tif'], output_file), 
                           os.path.join(local_dir_paths['initial_tiff'], file_name), 
                           format="GTiff", 
                           srcWin=[i, j, w, h])


def split_tiff_geo(opt, local_dir_paths, file_name,location_id, file_ind):
    """
    Break down initial large tiff files into smaller tif tiles
    """
            
    dset = gdal.Open(os.path.join(local_dir_paths['initial_tiff'], file_name))
    geotransform = dset.GetGeoTransform()
    left_x = geotransform[0]
    top_y = geotransform[3]
    #find the image width and height in meters since the projection comes in transverse mercator
    width = dset.RasterXSize*geotransform[1]
    height = dset.RasterYSize*geotransform[3] #negative value!
    
    tilewidth = opt['tilewidth_meters']
    tileheight = opt['tileheight_meters']

    # deglen = 110.25*1000 # distance of one degree at equator in m
    # tileheight = -tileheight/deglen #negative value to deal with top left anchor
    # tilewidth = tilewidth/(deglen*np.cos(left_x)) #cos of latitude accounts for difference in degree length at different latitudes
    
    ###TODO: fix this issue: Planet sometimes sends us absurdly large tiles that aren't clipped, skipping for now
    if width > 10*tilewidth or height > 10*tileheight:
        print('Warning: this image has more than 10 tiles per side, skipping')
        return 
    for i in tqdm(range(0, width, tilewidth), desc='Splitting'):
        for j in range(0, height, tileheight):
            w = min(i + tilewidth, width) - i
            h = max(j + tileheight, height) - j
            output_file = '{}_{}_{}_{}.tif'.format(location_id, file_ind, i, j)
            # Create smaller tif file of subregion
            
            
            gdal.Translate(os.path.join(local_dir_paths['tiled_tif'], output_file), 
                           os.path.join(local_dir_paths['initial_tiff'], file_name), 
                           format="GTiff", 
                           projwin=[left_x+i, top_y+j, left_x+i+w, top_y+j+h])


def process_initial_tiffs(bucket, opt, gc_dir_paths, local_dir_paths):
    """
    Goes Through all the steps to download, split up and save metadata of composite geotifs from planet
    """
    # see if we need to complete this step
    if check_previous_tiled_local(local_dir_paths['tiled_tif']):
        print("tiled_tifs exists locally and completed.txt is present")
        if opt['replace_tiled_local'] == 'ask':
            response = input("Do you want to start from scratch, wipe the directory, and create new tiles? [y/N]: ")
            if response != "y" and response != "Y":
                print("skipping")
                print()
                return
            else:
                print("re-tiling")
                print()
        elif opt['replace_tiled_local'] == 'all':
            print("re-tiling")
            print()
        elif opt['replace_tiled_local'] == 'skip':
            print("skipping")
            print()
            return
        else:
            raise Exception("Invalid option for parameter 'replace_tiled_local'. Must be one of 'all', 'ask', or 'skip'.")
        
    # create the directory if it doesn't exist
    if not os.path.isdir(local_dir_paths['tiled_tif']):
        os.makedirs(local_dir_paths['tiled_tif'])
    # wipe the directory if it exists
    else:
        shutil.rmtree(local_dir_paths['tiled_tif'])
        os.makedirs(local_dir_paths['tiled_tif'])
    
    # if tiled tifs exists in the cloud, see if the user wants to pull that down and not have to recreate
    if check_previous_tiled_gcloud(bucket, gc_dir_paths['tiled_tif']):
        print("tiled_tif already exists in the cloud")
        if opt['replace_tiled_cloud'] == 'ask':
            response = input("Do you want to 1. pull down the tiled_tif in the cloud or 2. create new tiled_tifs? [1/2]: ")
            while response != '1' and response != '2':
                print("Did not understand the response, please input either 1 or 2")
                response = input("Do you want to 1. pull down the tiled_tif in the cloud or 2. create new tiled_tifs? [1/2]: ")
                
            if response == 1:
                print("syncing with cloud")
                download_tiled_tifs(bucket, gc_dir_paths['tiled_tif'], local_dir_paths['tiled_tif'])
                return
                
            elif response == 2:
                print("recreating tiles")
        elif opt['replace_tiled_cloud'] == 'all':
            print('re-tiling')
            print()
        elif opt['replace_tiled_cloud'] == 'skip':
            print("syncing with cloud")
            download_tiled_tifs(bucket, gc_dir_paths['tiled_tif'], local_dir_paths['tiled_tif'])
            return
        else:
            raise Exception("Invalid option for parameter 'replace_tiled_cloud'. Must be one of 'all', 'ask', or 'skip'.")


    # pull the initial tiffs down from the cloud
    print("downloading initial tiffs for {}".format(gc_dir_paths['cafo_id']))
    download_initial_tiffs(bucket, gc_dir_paths['initial_tiff'], local_dir_paths['initial_tiff'])
    
    print("splitting tiff files into tiles")
            
    # split the initial tiffs into tiles
    tiff_list = os.listdir(local_dir_paths['initial_tiff'])
    for file_ind, a_tiff in enumerate(tiff_list):
        if a_tiff == 'composite.tif':
            # we are splitting the composite file specifically
            split_tiff(opt, local_dir_paths, a_tiff, local_dir_paths['cafo_id'], local_dir_paths['date_id'])
        
    print("removing initial tiffs")
    shutil.rmtree(local_dir_paths['initial_tiff'])
    
    with open(os.path.join(local_dir_paths['tiled_tif'], "completed.txt"), 'w') as f:
        pass
    

    if opt['gc_save']:
        print('upload to gcloud')
        upload_dir_to_gcloud(bucket, local_dir_paths['tiled_tif'], gc_dir_paths['tiled_tif'], info='tiled_tif')


def cut_dark(opt, local_dir_paths, verbose=True):
    """
    Determine how many black pixels a jpeg has, and subsequently move (remove) it if this is above a threshold
    """
    if verbose:
        print('Removing images with > ' + str(opt['black_thresh']*100), '% black pixels.')
        print('Moving to ...{}'.format(local_dir_paths['black_imgs']))

    all_jpg_files = [a_file for a_file in os.listdir(local_dir_paths['images']) if '.jpeg' in a_file]
    count = 0
    start = time.time()
    for filename in tqdm(all_jpg_files, disable=not verbose, desc="cutting dark images"):
        img = plt.imread(os.path.join(local_dir_paths['images'], filename))
        num_nblk = count_nonblack_np(img)
        tot_pixels = np.prod(img.shape[0:2], dtype=float)

        if (num_nblk / tot_pixels < 1-opt['black_thresh']):
            if os.path.exists(os.path.join(local_dir_paths['black_imgs'], filename)):
                os.remove(os.path.join(local_dir_paths['black_imgs'], filename))
            shutil.move(os.path.join(local_dir_paths['images'], filename), local_dir_paths['black_imgs'])
            count += 1
            
    with open(os.path.join(local_dir_paths['black_imgs'], "completed.txt"), 'w') as f:
        pass

    if verbose:
        print('Done. ({0:.3f} seconds.)'.format(time.time() - start))
        print('{} images moved.\n'.format(count))

def create_jpgs(opt, local_dir_paths, verbose=True):

    tif_names = [a_file for a_file in os.listdir(local_dir_paths['tiled_tif']) if '.tif' in a_file]
    if os.path.exists(local_dir_paths['images']):
        shutil.rmtree(local_dir_paths['images'])
        os.makedirs(local_dir_paths['images'])
    else:
        os.makedirs(local_dir_paths['images'])
    
    for tif_name in tqdm(tif_names, disable=not verbose, desc='constructing jpgs'):
        jpeg_name = tif_name.split(".tif")[0] + ".jpeg"
    
        # Create image
        gdal.Translate(os.path.join(local_dir_paths['images'], jpeg_name), 
                       os.path.join(local_dir_paths['tiled_tif'], tif_name), 
                       options='-ot Byte -of JPEG -colorinterp red,green,blue,alpha')
        # cleanup
        xmlFile = os.path.join(local_dir_paths['images'], tif_name.split(".tif")[0] + '.jpeg.aux.xml')
        os.remove(xmlFile)

    with open(os.path.join(local_dir_paths['images'], "completed.txt"), 'w') as f:
        pass

#-------------------------------------------------------------------------------------------------------------------------------
# 1. create the jpegs from the tiled_tifs if necessary
# 2. move (or remove) images for which more than the threshold of the pixels are black
#-------------------------------------------------------------------------------------------------------------------------------
def process_jpgs(bucket, opt, gc_dir_paths, local_dir_paths):
    
    # create jpegs from the tiled_tifs
    create_jpgs(opt, local_dir_paths)
    
    print("Saving tif coordinates")
    df = get_tif_coordinates(local_dir_paths['tiled_tif'])
    df.to_csv(Path(os.path.join(local_dir_paths['images'],'coordinates.csv')), index=False)

    # move (remove) images where more than the treshold of the pixels are black
    if opt['black_thresh'] is not None and opt['black_thresh'] > 0:
        cut_dark(opt, local_dir_paths)
        if opt['gc_save']:
            upload_dir_to_gcloud(bucket, local_dir_paths['black_imgs'], gc_dir_paths['black_imgs'], info='black_imgs')
            
    # push the jpegs to gcloud if desired
    if opt['gc_save']:
        upload_dir_to_gcloud(bucket, local_dir_paths['images'], gc_dir_paths['images'], info='images')
        
    # remove all local files if desired
    if not opt['local_save']:
        print('Removing local files')
        shutil.rmtree(local_dir_paths['root'])

def main(bucket, opt, current_folder):
    """
    Handle preprocessing.
    Args:
        bucket: gcs bucket object
        opt: config file
        current_folder: name of start_date folder for this run

    The final tiled images are stored under {dataset}_{loc_id}
    """


    # paths to image blobs are then: {dataset}/{loc_id}/{start_date}/images/{blah}.{extension}

    logging.info('searching bucket for downloaded images')
    files_gen = bucket.list_blobs(prefix=add_sep(opt['gcs_save_path']))
    split_names = [file.name.split('/') for file in files_gen]
    logging.info(f"Looking at folder: {current_folder}")
    split_names = [split_name for split_name in split_names if current_folder in split_name ]
    initial_images = [split_name for split_name in split_names if 'composite.tif' in split_name]
    logging.debug(f"Found following images to process: {initial_images}")

    # get cafo folders from initial tif image paths (minus the file name)
    order_folder_names = list(set(["/".join(split_name[:-1]) + '/' for split_name in initial_images]))
    dataset_folder_names = ["/".join(name.split('/')[:-2]) + '/' for name in order_folder_names]

    # main loop
    for i in tqdm(range(len(order_folder_names)), total=len(order_folder_names), desc='processing locations'):
        order_folder = order_folder_names[i]
        dataset_folder = dataset_folder_names[i]
        
        # split initial tiffs into tiled tifs and upload back to cloud if desired
        logging.info("preprocessing: {}".format(dataset_folder))
        
        gc_dir_paths = {}
        gc_dir_paths['cafo_id'] = dataset_folder.split("/")[-3]
        gc_dir_paths['date_id'] = dataset_folder.split("/")[-2]
        gc_dir_paths['initial_tiff'] = order_folder
        gc_dir_paths['tiled_tif'] = os.path.join(dataset_folder, "tiled_tif")
        gc_dir_paths['images'] = os.path.join(dataset_folder, "images")
        gc_dir_paths['black_imgs'] = os.path.join(dataset_folder, "black_imgs")

        local_dir_paths = {}
        local_dir_paths['cafo_id'] = dataset_folder.split("/")[-3]
        local_dir_paths['date_id'] = dataset_folder.split("/")[-2]
        local_dir_paths['initial_tiff'] = os.path.join(opt['root'], dataset_folder, "initial_tiff")
        local_dir_paths['tiled_tif'] = os.path.join(opt['root'], dataset_folder, "tiled_tif")
        local_dir_paths['images'] = os.path.join(opt['root'], dataset_folder, "images")
        local_dir_paths['black_imgs'] = os.path.join(opt['root'], dataset_folder, "black_imgs")

        logging.debug(f"Local directories for reading/writing: {local_dir_paths}")
        for key, val in local_dir_paths.items():
            if key == 'cafo_id' or key == 'date_id':
                continue
            
            if not os.path.isdir(val):
                os.makedirs(val)

        process_initial_tiffs(bucket, opt, gc_dir_paths, local_dir_paths)
        process_jpgs(bucket, opt, gc_dir_paths, local_dir_paths)
