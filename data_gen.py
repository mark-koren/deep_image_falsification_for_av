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
import pdb


mypath = '/home/mark/Research/cs236project/deep_image_falsification_for_av/training_0000'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for filename in onlyfiles:
    if 'tfrecord' not in filename:
        continue
    dataset = tf.data.TFRecordDataset(mypath + '/' + filename, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        methods = [tf.compat.v2.image.ResizeMethod.AREA,
                   tf.compat.v2.image.ResizeMethod.BICUBIC,
                   tf.compat.v2.image.ResizeMethod.BILINEAR, 
                   #tf.compat.v2.image.ResizeMethod.GAUSSIAN,
                   # tf.compat.v2.image.ResizeMethod.LANCZOS3, 
                   # tf.compat.v2.image.ResizeMethod.LANCZOS5,
                   # tf.compat.v2.image.ResizeMethod.MITCHELLCUBIC, 
                   tf.compat.v2.image.ResizeMethod.NEAREST_NEIGHBOR]

        img =  tf.image.decode_jpeg(frame.images[0].image)[:, int((1920-1280)/2):int((1920-1280)/2+1280), :]
        plt.figure(figsize=(25, 20))
        plt.imshow(img)
        plt.show()
        plt.figure(figsize=(25, 20))
        for index, method in enumerate(methods):
            print(index, method)
            layout = [2, 2, index+1]
            ax = plt.subplot(*layout)
            img_small = tf.image.resize(img, (128,128), method=method)
            plt.imshow(img_small, cmap=None)
            # plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
            plt.grid(False)
            plt.axis('off')
        plt.show()
        pdb.set_trace()

        
        # img_small = tf.image.resize(img, (32,32), method=ResizeMethod.BILINEAR)
        # plot_image(img_small)
        
        

pdb.set_trace()
print("------- end of script --------")