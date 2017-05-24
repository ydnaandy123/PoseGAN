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


pp = pprint.PrettyPrinter()


get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def config_check(flags, default_setting=False):

    if default_setting:
        if flags.name.find('cityscapes') != -1:
            flags.dataset_name = 'CITYSCAPES_DATASET'
            if flags.name.find('semantic') != -1:
                flags.image_dir = './dataset/CITYSCAPES_DATASET/train/semantic_color'
                flags.condition_dir = './dataset/CITYSCAPES_DATASET/train/image'
                flags.input_fname_pattern = '*.png'
                flags.image_height = 256
                flags.image_width = 512
                flags.image_dim = 3
                flags.condition_height = 256
                flags.condition_width = 512
                flags.condition_dim = 3
            if flags.name.find('image') != -1:
                flags.image_dir = './dataset/CITYSCAPES_DATASET/train/image'
                flags.condition_dir = './dataset/CITYSCAPES_DATASET/train/semantic_color'
                flags.input_fname_pattern = '*.png'
                flags.image_height = 256
                flags.image_width = 512
                flags.image_dim = 3
                flags.condition_height = 256
                flags.condition_width = 512
                flags.condition_dim = 3
        elif flags.name.find('mpii') != -1:
            flags.dataset_name = 'MPII'
            if flags.name.find('heatmap') != -1:
                flags.image_dir = './dataset/MPII/train/annot'
                flags.condition_dir = './dataset/MPII/train/images'
                flags.input_fname_pattern = '*.png'
                flags.image_height = 256
                flags.image_width = 256
                flags.image_dim = 3
                flags.condition_height = 256
                flags.condition_width = 256
                flags.condition_dim = 3

        flags.need_condition = (flags.name.find('(y)condition') != -1)
        flags.need_g1 = (flags.name.find('(y)g1') != -1)

    flags.test_dir = flags.name + '_' + flags.test_dir
    flags.sample_dir = flags.name + '_' + flags.sample_dir

    try:
        if flags.image_width is None:
            flags.image_width = flags.image_height
        if flags.condition_width is None:
            flags.condition_width = flags.condition_height
        model_dir = "{}_{}_{}_{}".format(flags.name, flags.batch_size, flags.image_height, flags.image_width)
    except AttributeError:
        if flags.input_width is None:
            flags.input_width = flags.input_height
        if flags.output_width is None:
            flags.output_width = flags.output_height
        model_dir = "{}_{}_{}_{}".format(flags.name, flags.batch_size, flags.output_height, flags.output_width)

    if not os.path.exists(flags.checkpoint_dir):
        os.makedirs(flags.checkpoint_dir)
    if not os.path.exists(flags.sample_dir):
        os.makedirs(flags.sample_dir)

    return flags, model_dir


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_image_condition(files, condition_dir, need_flip=True):
    images, conditions = [], []
    for name_idx, name_file in enumerate(files):
        # pose heatmap
        name = name_file.split('/')[-1]
        image = scipy.misc.imread(name_file).astype(np.float32)[:, :, :3]
        condition = scipy.misc.imread(os.path.join(condition_dir, name)).astype(np.float32)[:, :, :3]
        if need_flip and name_idx > len(files) / 2:
            image = np.fliplr(image)
            condition = np.fliplr(condition)
        images.append(image)
        conditions.append(condition)

    images = np.array(images).astype(np.float32) / 127.5 - 1.
    conditions = np.array(conditions).astype(np.float32) / 127.5 - 1.
    return images, conditions


def get_image_condition_pose(files, condition_dir, pose_num=16):
    images, conditions = [], []
    for name_file in files:
        # pose heatmap
        image = scipy.misc.imread(name_file).astype(np.float32)
        heatmap_joints = np.zeros((64, 64, pose_num)).astype(np.float32)
        for pose_idx in range(0, pose_num):
            heatmap_joint = np.zeros((64, 64), np.float32)
            cord_y, cord_x = np.nonzero(image == (pose_idx + 1))
            #if pose_idx == 6 or pose_idx == 7 or len(cord[0]) == 0:
            #    continue
            if len(cord_y) == 0:
                continue
            heatmap_joint[cord_y, cord_x] = 1
            blurred = gaussian_filter(heatmap_joint, sigma=3)
            blurred /= np.max(blurred)

            heatmap_joints[:, :, pose_idx] = blurred
        heatmap_joints = (heatmap_joints * 2.) - 1.
        images.append(heatmap_joints)

        name = name_file.split('/')[-1]
        condition = simple_get_image(os.path.join(condition_dir, name))
        conditions.append(condition)

    return np.array(images).astype(np.float32), np.array(conditions).astype(np.float32)


def get_image_condition_pose_mpii(files, condition_dir, channel_num=29):
    """
    :param files: image_dir
    :param condition_dir: heatmap_dir (64x64)
    :param channel_num: num of channels (max:16+13)
    :return: image and heatmap(64x64x(16+13)), normalized in -1~1
    """
    images, conditions = [], []
    part_array = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14]
    for name_file in files:
        # pose heatmap
        image = scipy.misc.imread(name_file).astype(np.float32)
        heatmap = np.zeros((64, 64, channel_num)).astype(np.float32)
        cur_channel = 0
        # every joints and limbs
        for pose_idx in range(0, 16):
            # joints
            heatmap_joint = np.zeros((64, 64), np.float32)
            cord_y, cord_x = np.nonzero(image == (pose_idx + 1))
            if len(cord_y) != 0:
                heatmap_joint[cord_y, cord_x] = 1
                blurred = gaussian_filter(heatmap_joint, sigma=3)
                blurred /= np.max(blurred)
                heatmap[:, :, cur_channel] = blurred
            cur_channel += 1
            if cur_channel == channel_num:
                break
            # limbs
            if pose_idx in part_array:
                heatmap_limb = np.zeros((64, 64), np.float32)
                cord_y_end, cord_x_end = np.nonzero(image == (pose_idx + 2))
                if len(cord_y) != 0 and len(cord_y_end) != 0:
                    rr, cc = line(cord_y, cord_x, cord_y_end, cord_x_end)
                    heatmap_limb[rr, cc] = 1
                    blurred = gaussian_filter(heatmap_limb, sigma=2)
                    blurred /= np.max(blurred)
                    heatmap[:, :, cur_channel] = blurred
                cur_channel += 1
                if cur_channel == channel_num:
                    break
        heatmap = (heatmap * 2.) - 1.
        images.append(heatmap)

        name = name_file.split('/')[-1]
        condition = simple_get_image(os.path.join(condition_dir, name))
        conditions.append(condition)

    return np.array(images).astype(np.float32), np.array(conditions).astype(np.float32)


def get_image_condition_pose_mpii_big(files, condition_dir, channel_num=29):
    """
    :param files: image_dir
    :param condition_dir: heatmap_dir (64x64)
    :param channel_num: num of channels (max:16+13)
    :return: image and heatmap(64x64x(16+13)), normalized in -1~1
    """
    images, conditions = [], []
    part_array = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14]
    for name_file in files:
        # pose heatmap
        image = scipy.misc.imread(name_file).astype(np.float32)
        heatmap = np.zeros((256, 256, channel_num)).astype(np.float32)
        cur_channel = 0
        # every joints and limbs
        for pose_idx in range(0, 16):
            # joints
            heatmap_joint = np.zeros((256, 256), np.float32)
            cord_y, cord_x = np.nonzero(image == (pose_idx + 1))
            if len(cord_y) != 0:
                heatmap_joint[cord_y * 4, cord_x * 4] = 1
                blurred = gaussian_filter(heatmap_joint, sigma=12)
                blurred /= np.max(blurred)
                heatmap[:, :, cur_channel] = blurred
            cur_channel += 1
            if cur_channel == channel_num:
                break
            # limbs
            if pose_idx in part_array:
                heatmap_limb = np.zeros((256, 256), np.float32)
                cord_y_end, cord_x_end = np.nonzero(image == (pose_idx + 2))
                if len(cord_y) != 0 and len(cord_y_end) != 0:
                    rr, cc = line(cord_y * 4, cord_x * 4,
                                  cord_y_end * 4, cord_x_end * 4)
                    heatmap_limb[rr, cc] = 1
                    blurred = gaussian_filter(heatmap_limb, sigma=8)
                    blurred /= np.max(blurred)
                    heatmap[:, :, cur_channel] = blurred
                cur_channel += 1
                if cur_channel == channel_num:
                    break
        heatmap = (heatmap * 2.) - 1.
        images.append(heatmap)

        name = name_file.split('/')[-1]
        condition = simple_get_image(os.path.join(condition_dir, name))
        conditions.append(condition)

    return np.array(images).astype(np.float32), np.array(conditions).astype(np.float32)


def heatmap_visual(heatmaps):
    heatmap_vs = []
    for heatmap in heatmaps:
        heatmap_v = np.zeros((64, 64, 3), dtype=np.float32)
        for pose_idx in [0, 1, 2, 10, 11, 12]:
            heatmap_v[:, :, 0] += heatmap[:, :, pose_idx]
        for pose_idx in [3, 4, 5, 13, 14, 15]:
            heatmap_v[:, :, 2] += heatmap[:, :, pose_idx]
        for pose_idx in [6, 7, 8, 9]:
            heatmap_v[:, :, 1] += heatmap[:, :, pose_idx]

        heatmap_v[np.nonzero(heatmap_v > 255.)] = 255.
        heatmap_vs.append(heatmap_v)

    return np.array(heatmap_vs).astype(np.float32)


def heatmap_visual_mpii(heatmaps):
    """
    :param heatmaps: (w*h*29) 0~255
    :return: (w*h*3) 0~255
    """
    h, w = heatmaps.shape[1], heatmaps.shape[2]
    heatmap_vs = []
    for heatmap in heatmaps:
        heatmap_v = np.zeros((h, w, 3), dtype=np.float32)
        # right legs
        for pose_idx in range(0, 5):
            heatmap_v[:, :, 0] += heatmap[:, :, pose_idx] * 0.8
        # left legs
        for pose_idx in range(6, 11):
            heatmap_v[:, :, 2] += heatmap[:, :, pose_idx] * 0.8
        # trunk
        for pose_idx in range(11, 18):
            heatmap_v[:, :, 1] += heatmap[:, :, pose_idx] * 0.8
        # right arm
        for pose_idx in range(18, 23):
            heatmap_v[:, :, 0] += heatmap[:, :, pose_idx] * 0.8
        # left arm
        for pose_idx in range(24, 29):
            heatmap_v[:, :, 2] += heatmap[:, :, pose_idx] * 0.8

        heatmap_v[:, :, 1] += heatmap[:, :, 5]
        heatmap_v[:, :, 1] += heatmap[:, :, 23]

        heatmap_v[np.nonzero(heatmap_v > 255.)] = 255.
        heatmap_vs.append(heatmap_v)

    return np.array(heatmap_vs).astype(np.float32)


def heatmap_visual_mpii_big(heatmaps):
    """
    :param heatmaps: (w*h*29) 0~255
    :return: (w*h*3) 0~255
    """
    heatmap_vs = []
    for heatmap in heatmaps:
        heatmap_v = np.zeros((64, 64, 3), dtype=np.float32)
        # right legs
        for pose_idx in range(0, 5):
            heatmap_v[:, :, 0] += heatmap[:, :, pose_idx] * 0.8
        # left legs
        for pose_idx in range(6, 11):
            heatmap_v[:, :, 2] += heatmap[:, :, pose_idx] * 0.8
        # trunk
        for pose_idx in range(11, 18):
            heatmap_v[:, :, 1] += heatmap[:, :, pose_idx] * 0.8
        # right arm
        for pose_idx in range(18, 23):
            heatmap_v[:, :, 0] += heatmap[:, :, pose_idx] * 0.8
        # left arm
        for pose_idx in range(24, 29):
            heatmap_v[:, :, 2] += heatmap[:, :, pose_idx] * 0.8

        heatmap_v[:, :, 1] += heatmap[:, :, 5]
        heatmap_v[:, :, 1] += heatmap[:, :, 23]

        heatmap_v[np.nonzero(heatmap_v > 255.)] = 255.
        heatmap_vs.append(heatmap_v)

    return np.array(heatmap_vs).astype(np.float32)


def denorm_image(image):
    return (image + 1.) * 127.5


def norm_image(image):
    return (image / 127.5) - 1.


def heatmap_visual_sig(heatmaps):
    heatmap_vs = []
    heatmaps = heatmaps * 255.
    for heatmap in heatmaps:
        heatmap_v = np.zeros((64, 64, 3), dtype=np.float32)
        for pose_idx in [0, 1, 2, 10, 11, 12]:
            heatmap_v[:, :, 0] += heatmap[:, :, pose_idx]
        for pose_idx in [3, 4, 5, 13, 14, 15]:
            heatmap_v[:, :, 2] += heatmap[:, :, pose_idx]
        for pose_idx in [6, 7, 8, 9]:
            heatmap_v[:, :, 1] += heatmap[:, :, pose_idx]

        heatmap_v[np.nonzero(heatmap_v > 255.)] = 255.
        heatmap_vs.append(heatmap_v)

    return np.array(heatmap_vs).astype(np.float32)


def simple_get_image(path):
    img = scipy.misc.imread(path).astype(np.float)
    return np.array(img) / 127.5 - 1.


def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
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


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(
        x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(
          image, input_height, input_width,
          resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                  var layer_%s = {
                    "layer_type": "fc",
                    "sy": 1, "sx": 1,
                    "out_sx": 1, "out_sy": 1,
                    "stride": 1, "pad": 0,
                    "out_depth": %s, "in_depth": %s,
                    "biases": %s,
                    "gamma": %s,
                    "beta": %s,
                    "filters": %s
                  };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                  var layer_%s = {
                    "layer_type": "deconv",
                    "sy": 5, "sx": 5,
                    "out_sx": %s, "out_sy": %s,
                    "stride": 2, "pad": 1,
                    "out_depth": %s, "in_depth": %s,
                    "biases": %s,
                    "gamma": %s,
                    "beta": %s,
                    "filters": %s
                  };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                       W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
    image_frame_dim = int(math.ceil(config.batch_size**.5))
    if option == 0:
        z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
    elif option == 1:
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in range(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            if config.dataset == "mnist":
                y = np.random.choice(10, config.batch_size)
                y_one_hot = np.zeros((config.batch_size, 10))
                y_one_hot[np.arange(config.batch_size), y] = 1

                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
            else:
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

            save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))
    elif option == 2:
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in [random.randint(0, 99) for _ in range(100)]:
            print(" [*] %d" % idx)
            z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
            z_sample = np.tile(z, (config.batch_size, 1))
            #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            if config.dataset == "mnist":
                y = np.random.choice(10, config.batch_size)
                y_one_hot = np.zeros((config.batch_size, 10))
                y_one_hot[np.arange(config.batch_size), y] = 1

                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
            else:
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

            try:
                make_gif(samples, './samples/test_gif_%s.gif' % (idx))
            except:
                save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
    elif option == 3:
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in range(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    elif option == 4:
        image_set = []
        values = np.arange(0, 1, 1./config.batch_size)

        for idx in range(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

            image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
            make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

        new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
            for idx in range(64) + range(63, -1, -1)]
        make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)
