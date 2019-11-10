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

def show_camera_image(camera_image, camera_labels, layout, cmap=None):
    """Show a camera image and the given camera labels."""
    # pdb.set_trace()
    ax = plt.subplot(*layout)

    # Draw the camera labels.
    # for camera_labels in frame.camera_labels:
    #   # Ignore camera labels that do not correspond to this camera.
    #   if camera_labels.name != camera_image.name:
    #     continue

    #   # Iterate over the individual labels.
    #   for label in camera_labels.labels:
    #     # Draw the object bounding box.
    #     ax.add_patch(patches.Rectangle(
    #       xy=(label.box.center_x - 0.5 * label.box.length,
    #           label.box.center_y - 0.5 * label.box.width),
    #       width=label.box.length,
    #       height=label.box.width,
    #       linewidth=1,
    #       edgecolor='red',
    #       facecolor='none'))

    # Show the camera image.

    # plt.figure(figsize=(20, 12))
    # plt.imshow(tf.image.decode_jpeg(camera_image.image))
    # plt.grid("off")

    plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
    plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
    plt.grid(False)
    plt.axis('off')



if __name__ == "__main__":
    FILENAME = './training_0000/segment-1208303279778032257_1360_000_1380_000_with_camera_labels.tfrecord'
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        break

    plt.figure(figsize=(25, 20))
    # plt.show()

    for index, image in enumerate(frame.images):
        show_camera_image(image, frame.camera_labels, [3, 3, index+1])

    plt.show()