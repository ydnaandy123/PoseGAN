"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from skimage.draw import line
from scipy.ndimage.filters import gaussian_filter
cityscapes_labels = [
    (  0,  0,  0),
    (  0,  0,  0),
    (  0,  0,  0),
    (  0,  0,  0),
    (  0,  0,  0),
    (111, 74,  0),
    ( 81,  0, 81),
    (128, 64,128),
    (244, 35,232),
    (250,170,160),
    (230,150,140),
    ( 70, 70, 70),
    (102,102,156),
    (190,153,153),
    (180,165,180),
    (150,100,100),
    (150,120, 90),
    (153,153,153),
    (153,153,153),
    (250,170, 30),
    (220,220,  0),
    (107,142, 35),
    (152,251,152),
    ( 70,130,180),
    (220, 20, 60),
    (255,  0,  0),
    (  0,  0,142),
    (  0,  0, 70),
    (  0, 60,100),
    (  0,  0, 90),
    (  0,  0,110),
    (  0, 80,100),
    (  0,  0,230),
    (119, 11, 32),
    (  0,  0,142)
]


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))
# ===================================================
# -----------------Main-------------------------
# ===================================================
pp = pprint.PrettyPrinter()


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def config_check_directory(flags):
    flags.test_dir = flags.name + '_' + flags.test_dir
    flags.sample_dir = flags.name + '_' + flags.sample_dir
    if not os.path.exists(flags.checkpoint_dir):
        os.makedirs(flags.checkpoint_dir)
    if not os.path.exists(flags.sample_dir):
        os.makedirs(flags.sample_dir)


def overrides(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider


# ===================================================
# -----------------Data processing-------------------
# ===================================================
def read_image(path, dim=3):
    if dim == 0:
        return scipy.misc.imread(path).astype(np.uint8)
    else:
        return scipy.misc.imread(path).astype(np.uint8)[:, :, :dim]


def norm_image(img):
    return img.astype(np.float32) / 127.5 - 1.0


def denorm_image(img):
    return (img.astype(np.float32) + 1.0) * 127.5


def get_batch_images_norm(batch, need_flip=True, scale=1.0):
    tensor = []
    for path_idx, path in enumerate(batch):
        # np.uint8
        # image read, flip, resize
        img = read_image(path)
        if need_flip and path_idx > len(batch) / 2:
            img = np.fliplr(img)
        if scale != 1.0:
            img = scipy.misc.imresize(img, scale)
        # np.float32
        tensor.append(norm_image(img))

    return np.array(tensor).astype(np.float32)


def get_batch_images(batch, need_flip=True, scale=1.0, dim=3):
    tensor = []
    for path_idx, path in enumerate(batch):
        # np.uint8
        # image read, flip, resize
        img = read_image(path, dim)
        if need_flip and path_idx > len(batch) / 2:
            img = np.fliplr(img)
        if scale != 1.0:
            img = scipy.misc.imresize(img, scale)
        # np.float32
        tensor.append(img)

    return np.array(tensor).astype(np.float32)

# ===================================================
# -----------------Inpainting--------------------
# ===================================================
def generate_batch_holes_norm(batch, mask_dir, need_flip=True, scale=1.0):
    tensor = []
    for path_idx, path in enumerate(batch):
        # np.uint8
        # image read, flip, resize
        img = read_image(path)
        if need_flip and path_idx > len(batch) / 2:
            img = np.fliplr(img)
        if scale != 1.0:
            img = scipy.misc.imresize(img, scale)
        # np.float32
        tensor.append(norm_image(img))

    return np.array(tensor).astype(np.float32)





# ===================================================
# -----------------Data Visualize--------------------
# ===================================================

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3,4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def label_visual(label_batchs):
    label_batchs = np.array(np.argmax(label_batchs, axis=3))
    w, h, = label_batchs[0].shape[0], label_batchs[0].shape[1]
    visuals = []
    for label in label_batchs:
        visual = np.zeros((w, h, 3), dtype=np.float32)
        for i in range(0, 34):
            index = np.nonzero(label == i)
            visual[index + (0,)] = cityscapes_labels[i][0]
            visual[index + (1,)] = cityscapes_labels[i][1]
            visual[index + (2,)] = cityscapes_labels[i][2]
        visuals.append(visual)
    return np.array(visuals).astype(np.float32)