from os import listdir
from os.path import isfile, join
import cv2

import os
import tensorflow as tf
import math
import numpy as np
import itertools
print(tf.__version__)
tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import pdb


def image_test():
    mypath = '/home/mkoren/scratch/Research/cs236project/deep_image_falsification_for_av/training_0000'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for filename in onlyfiles:
        if 'tfrecord' not in filename:
            continue
        dataset = tf.data.TFRecordDataset(mypath + '/' + filename, compression_type='')
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            methods = [Image.NEAREST,
                       Image.BILINEAR,
                       Image.BICUBIC,
                       Image.LANCZOS,
                       Image.HAMMING,
                       Image.BOX]
            # methods = [tf.compat.v2.image.ResizeMethod.AREA,
                     # tf.compat.v2.image.ResizeMethod.BICUBIC,
                     # tf.compat.v2.image.ResizeMethod.BILINEAR, 
                     #tf.compat.v2.image.ResizeMethod.GAUSSIAN,
                     # tf.compat.v2.image.ResizeMethod.LANCZOS3, 
                     # tf.compat.v2.image.ResizeMethod.LANCZOS5,
                     # tf.compat.v2.image.ResizeMethod.MITCHELLCUBIC, 
                     # tf.compat.v2.image.ResizeMethod.NEAREST_NEIGHBOR]

            # img =  tf.image.decode_jpeg(frame.images[0].image)[:, int((1920-1280)/2):int((1920-1280)/2+1280), :]
            img = Image.fromarray(tf.image.decode_jpeg(frame.images[0].image).numpy())
            plt.figure(figsize=(25, 20))
            plt.imshow(img)
            plt.show()
            plt.figure(figsize=(25, 20))
            for index, method in enumerate(methods):
                print(index, method)
                layout = [2, 3, index+1]
                ax = plt.subplot(*layout)
                # pdb.set_trace()
                img_small = img.resize((int(192/2),int(128/2)),resample=method)
                # img_small = tf.image.resize(img, (128,192), method=method)
                plt.imshow(img_small, cmap=None)
                # plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
                plt.grid(False)
                plt.axis('off')
            plt.show()
            pdb.set_trace()

def generate_dataset_pixelcnn(path_to_trainings, path_to_data):
    for dirname in os.listdir(path_to_trainings):
        if 'training_' in dirname:
            data_type = '/training'
        elif 'validation_' in dirname:
            data_type = '/test'
        else:
            continue
        print('starting directory: ', dirname)
        path_to_tfrecords = path_to_trainings + '/' + dirname
        onlyfiles = [f for f in listdir(path_to_tfrecords) if isfile(join(path_to_tfrecords, f))] 

        for record_index, filename in enumerate(onlyfiles):
            if 'tfrecord' not in filename:
                continue
            print('starting file: ', filename)
            dataset = tf.data.TFRecordDataset(path_to_tfrecords + '/' + filename, compression_type='')
            for img_index, data in enumerate(dataset):
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                img = Image.fromarray(tf.image.decode_jpeg(frame.images[0].image).numpy())

                img_small = img.resize((int(192/2),int(128/2)),resample=Image.LANCZOS)
                img_small.save(path_to_data + data_type + '/' + dirname + '_' + str(record_index) + '_' + str(img_index) + '.jpg')
                # pdb.set_trace()
    # pdb.set_trace()
        
        # img_small = tf.image.resize(img, (32,32), method=ResizeMethod.BILINEAR)
        # plot_image(img_small)

def generate_dataset_stargan(path_to_labels, path_to_images, path_to_data, max_image_count = np.inf):
    with open(join(path_to_labels, 'bdd100k_labels_images_train.json')) as f:
        data = json.load(f)
    
    for img_index, img_data in enumerate(data):
        if img_index >= max_image_count:
            break

        img_path = join(path_to_images, 'train', img_data['name'])
        with Image.open(img_path) as img:
            img_small = img.resize((128,72),resample=Image.LANCZOS)

            weather_type = img_data['attributes']['weather']
            time_of_day = img_data['attributes']['timeofday']

            if not os.path.exists(join(path_to_data, 'train', weather_type)):
                os.makedirs(join(path_to_data, 'train', weather_type))
            img_small.save(join(path_to_data, 'train', weather_type, img_data['name']))

            if not os.path.exists(join(path_to_data, 'train', time_of_day)):
                os.makedirs(join(path_to_data, 'train', time_of_day))
            img_small.save(join(path_to_data, 'train', time_of_day, img_data['name']))

    with open(join(path_to_labels, 'bdd100k_labels_images_val.json')) as f:
        data = json.load(f)
    
    for img_index, img_data in enumerate(data):
        if img_index >= max_image_count:
            break

        img_path = join(path_to_images, 'val', img_data['name'])
        with Image.open(img_path) as img:
            img_small = img.resize((128,72),resample=Image.LANCZOS)

            weather_type = img_data['attributes']['weather']
            time_of_day = img_data['attributes']['timeofday']

            if not os.path.exists(join(path_to_data, 'test', weather_type)):
                os.makedirs(join(path_to_data, 'test', weather_type))
            img_small.save(join(path_to_data, 'test', weather_type, img_data['name']))

            if not os.path.exists(join(path_to_data, 'test', time_of_day)):
                os.makedirs(join(path_to_data, 'test', time_of_day))
            img_small.save(join(path_to_data, 'test', time_of_day, img_data['name']))




        
# image_test()
# generate_dataset('/home/mkoren/scratch/Research/cs236project/deep_image_falsification_for_av',
#     '/home/mkoren/scratch/Research/cs236project/data/av_96_64')
# '/home/mkoren/Downloads/bdd100k/labels/bdd100k_labels_images_train.json'
generate_dataset_stargan(path_to_labels = '/home/mkoren/scratch/Research/cs236project/data/bdd100k/labels',
                         path_to_images = '/home/mkoren/scratch/Research/cs236project/data/bdd100k/images/100k',
                         path_to_data = '/home/mkoren/scratch/Research/cs236project/data/bdd100k_stargan',
                         # max_image_count = 100)
                         )
# pdb.set_trace()
print("------- end of script --------")