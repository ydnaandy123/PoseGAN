from glob import glob
import numpy as np
import os
import scipy.misc
import h5py
from scipy.ndimage.filters import gaussian_filter
import os.path
from skimage.draw import line


def crop_images(dataset_dir):
    """
    Read all images under the different folders
    Crop, resize and store them
    example code:
        crop_images(CITYSCAPES_dir)
    """
    data = []
    for folder in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, folder, "*_labelIds.png")
        data.extend(glob(path))

    for index, filePath in enumerate(data):
        print ('{}/{}'.format(index, len(data)))
        name = filePath.split('/')[-1].split('_')
        name_to_stroe = '{}_{}_{}.png'.format(name[0], name[1], name[2])
        img = scipy.misc.imread(filePath).astype(np.uint8)
        #img = scipy.misc.imresize(img, 0.125, interp='bilinear', mode=None)
        #img = scipy.misc.imresize(img, 0.125, interp='nearest', mode=None)
        scipy.misc.imsave('/data/vllab1/dataset/CITYSCAPES/CITYSCAPES_DATASET/cityscapesScripts/results/' + name_to_stroe, img)
        #break


def crop_images_same_dir(data_set_dir):
    """
    Read all images under the same folder
    Crop, resize and store them
    """
    data = glob(os.path.join(data_set_dir, "*_labelIds.png"))

    for index, filePath in enumerate(data):
        print ('%d/%d' % (index, len(data)))

        img = scipy.misc.imread(filePath).astype(np.uint8)
        #img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
        #scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_random/' + filePath.split('/')[-1],
        #                  img[offs_h[index]:offs_h_end[index], offs_w[index]:offs_w_end[index] :])
        scipy.misc.imsave('/data/vllab1/dataset/CITYSCAPES/CITYSCAPES_DATASET/cityscapesScripts/results/' + filePath.split('/')[-1],
                          img)
        break


def crop_lsp(data_set_dir):
    """
    """
    # All metedata
    filename = '/data/vllab1/pose-hg-train/data/LSP/annot/lsp.h5'
    f = h5py.File(filename, 'r')
    part = f['part']
    center = f['center']
    scale = f['scale']
    # train images
    for idx in range(1, 1000 + 1):
        print('{:d}/{:d}'.format(idx, 1000))
        im_name = 'im{:04d}.jpg'.format(idx)
        img = scipy.misc.imread(os.path.join(data_set_dir, im_name)).astype(np.float32)
        # Calculate region
        img_height, img_width, _ = img.shape
        h5py_idx = idx + 10000
        long_side = int(np.ceil(scale[h5py_idx - 1] * 100.))
        img_center_x, img_center_y = int(center[h5py_idx - 1][0]), int(center[h5py_idx - 1][1])
        pad_y_min, pad_x_min = img_center_y - long_side, img_center_x - long_side
        x_min, x_max = max(pad_x_min, 0), min(img_center_x + long_side, img_width)
        y_min, y_max = max(pad_y_min, 0), min(img_center_y + long_side, img_height)
        margin_y_min, margin_x = y_min - pad_y_min, x_min - pad_x_min
        # transplant image
        img_center = img[y_min:y_max, x_min:x_max, :]
        img_center_height, img_center_width, _ = img_center.shape
        img_pad = np.zeros((long_side*2, long_side*2, 3), dtype=np.float32)
        img_pad[margin_y_min:margin_y_min+img_center_height, margin_x:margin_x+img_center_width, :] = img_center
        img_pad = scipy.misc.imresize(img_pad, (256, 256), interp='bilinear', mode=None)
        # create heatmap
        heatmap = np.ones((64, 64), np.uint8) * 255
        for pose_idx in range(0, 16):
            cord_x, cord_y = int(part[h5py_idx-1][pose_idx][0]), int(part[h5py_idx-1][pose_idx][1])
            if pose_idx == 6 or pose_idx == 7 or cord_x < 0 or cord_y < 0:
                continue
            cord_y = max((cord_y - pad_y_min), 0) / float(long_side) * 32.
            cord_x = max((cord_x - pad_x_min), 0) / float(long_side) * 32.
            cord_y = min(cord_y, 63)
            cord_x = min(cord_x, 63)
            cord_y, cord_x = int(np.round(cord_y)), int(np.round(cord_x))
            heatmap[cord_y, cord_x] = (pose_idx + 1)

        # Output
        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/images', im_name.split('.')[0] + '.png'),
                          img_pad.astype(np.float32), format='png')
        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/annot', im_name.split('.')[0] + '.png'),
                          heatmap.astype(np.uint8), format='png')

    for idx in range(1, 10000 + 1):
        print('{:d}/{:d}'.format(idx, 10000))
        im_name = 'im{:05d}.jpg'.format(idx)
        img = scipy.misc.imread(os.path.join(data_set_dir, im_name)).astype(np.float32)
        # Calculate region
        img_height, img_width, _ = img.shape
        long_side = int(np.ceil(scale[idx - 1] * 100.))
        img_center_x, img_center_y = int(center[idx - 1][0]), int(center[idx - 1][1])
        pad_y_min, pad_x_min = img_center_y - long_side, img_center_x - long_side
        x_min, x_max = max(pad_x_min, 0), min(img_center_x + long_side, img_width)
        y_min, y_max = max(pad_y_min, 0), min(img_center_y + long_side, img_height)
        margin_y_min, margin_x = y_min - pad_y_min, x_min - pad_x_min
        # transplant image
        img_center = img[y_min:y_max, x_min:x_max, :]
        img_center_height, img_center_width, _ = img_center.shape
        img_pad = np.zeros((long_side*2, long_side*2, 3), dtype=np.float32)
        img_pad[margin_y_min:margin_y_min+img_center_height, margin_x:margin_x+img_center_width, :] = img_center
        img_pad = scipy.misc.imresize(img_pad, (256, 256), interp='bilinear', mode=None)
        # create heatmap
        heatmap = np.ones((64, 64), np.uint8) * 255
        for pose_idx in range(0, 16):
            cord_x, cord_y = int(part[idx-1][pose_idx][0]), int(part[idx-1][pose_idx][1])
            if pose_idx == 6 or pose_idx == 7 or cord_x < 0 or cord_y < 0:
                continue
            cord_y = max((cord_y - pad_y_min), 0) / float(long_side) * 32.
            cord_x = max((cord_x - pad_x_min), 0) / float(long_side) * 32.
            cord_y = min(cord_y, 63)
            cord_x = min(cord_x, 63)
            cord_y, cord_x = int(np.round(cord_y)), int(np.round(cord_x))
            heatmap[cord_y, cord_x] = (pose_idx + 1)

        # Output
        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/images', im_name.split('.')[0] + '.png'),
                          img_pad.astype(np.float32), format='png')
        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/annot', im_name.split('.')[0] + '.png'),
                          heatmap.astype(np.uint8), format='png')


def crop_lsp_test(data_set_dir):
    """
    """
    # All metedata
    filename = '/data/vllab1/pose-hg-train/data/LSP/annot/lsp.h5'
    f = h5py.File(filename, 'r')
    part = f['part']
    center = f['center']
    scale = f['scale']
    # train images
    for idx in range(1001, 2000 + 1):
        print('{:d}/{:d}'.format(idx, 1000))
        im_name = 'im{:04d}.jpg'.format(idx)
        img = scipy.misc.imread(os.path.join(data_set_dir, im_name)).astype(np.float32)
        # Calculate region
        img_height, img_width, _ = img.shape
        h5py_idx = idx + 10000
        long_side = int(np.ceil(scale[h5py_idx - 1] * 100.))
        img_center_x, img_center_y = int(center[h5py_idx - 1][0]), int(center[h5py_idx - 1][1])
        pad_y_min, pad_x_min = img_center_y - long_side, img_center_x - long_side
        x_min, x_max = max(pad_x_min, 0), min(img_center_x + long_side, img_width)
        y_min, y_max = max(pad_y_min, 0), min(img_center_y + long_side, img_height)
        margin_y_min, margin_x = y_min - pad_y_min, x_min - pad_x_min
        # transplant image
        img_center = img[y_min:y_max, x_min:x_max, :]
        img_center_height, img_center_width, _ = img_center.shape
        img_pad = np.zeros((long_side*2, long_side*2, 3), dtype=np.float32)
        img_pad[margin_y_min:margin_y_min+img_center_height, margin_x:margin_x+img_center_width, :] = img_center
        img_pad = scipy.misc.imresize(img_pad, (256, 256), interp='bilinear', mode=None)
        # create heatmap
        heatmap = np.ones((64, 64), np.uint8) * 255
        for pose_idx in range(0, 16):
            cord_x, cord_y = int(part[h5py_idx-1][pose_idx][0]), int(part[h5py_idx-1][pose_idx][1])
            if pose_idx == 6 or pose_idx == 7 or cord_x < 0 or cord_y < 0:
                continue
            cord_y = max((cord_y - pad_y_min), 0) / float(long_side) * 32.
            cord_x = max((cord_x - pad_x_min), 0) / float(long_side) * 32.
            cord_y = min(cord_y, 63)
            cord_x = min(cord_x, 63)
            cord_y, cord_x = int(np.round(cord_y)), int(np.round(cord_x))
            heatmap[cord_y, cord_x] = (pose_idx + 1)

        # Output
        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/LSP/test/images', im_name.split('.')[0] + '.png'),
                          img_pad.astype(np.float32), format='png')
        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/LSP/test/annot', im_name.split('.')[0] + '.png'),
                          heatmap.astype(np.uint8), format='png')


def read_crop_lsp():
    """
    """
    # All metedata
    filename = '/data/vllab1/pose-hg-train/data/LSP/annot/lsp.h5'
    f = h5py.File(filename, 'r')
    part = f['part']
    center = f['center']
    scale = f['scale']
    sig = 3
    # train images
    for idx in range(1, 1000 + 1):
        print('{:d}/{:d}'.format(idx, 11000))
        # Read data
        im_name = 'im{:04d}.png'.format(idx)
        img = scipy.misc.imread(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/images', im_name)).astype(
            np.float32)
        heatmap = scipy.misc.imread(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/annot', im_name)).astype(
            np.float32)
        # Parsing
        img_small = scipy.misc.imresize(img, (64, 64), interp='bilinear', mode=None).astype(np.float32)
        for pose_idx in range(0, 16):
            '''
            # Calculate region
            img_height, img_width, _ = img.shape
            long_side = int(np.ceil(scale[idx - 1] * 100.))
            img_center_x, img_center_y = int(center[idx - 1][0]), int(center[idx - 1][1])
            pad_y_min, pad_x_min = img_center_y - long_side, img_center_x - long_side
            x_min, x_max = max(pad_x_min, 0), min(img_center_x + long_side, img_width)
            y_min, y_max = max(pad_y_min, 0), min(img_center_y + long_side, img_height)
            margin_y_min, margin_x = y_min - pad_y_min, x_min - pad_x_min
            # create heatmap
            cord_x, cord_y = int(part[idx-1][pose_idx][0]), int(part[idx-1][pose_idx][1])
            if pose_idx == 6 or pose_idx == 7 or cord_x < 0 or cord_y < 0:
                continue
            cord_y = max((cord_y - pad_y_min), 0) / float(long_side) * 32.
            cord_x = max((cord_x - pad_x_min), 0) / float(long_side) * 32.
            cord_y = min(cord_y, 63)
            cord_x = min(cord_x, 63)
            cord_y, cord_x = int(np.round(cord_y)), int(np.round(cord_x))
            heatmap = np.zeros((64, 64), np.float32)
            heatmap[cord_y, cord_x] = 1
            '''

            heatmap_pose = np.zeros((64, 64), np.float32)
            cord = np.nonzero(heatmap == (pose_idx + 1))
            if pose_idx == 6 or pose_idx == 7 or len(cord[0]) == 0:
                continue
            heatmap_pose[cord] = 1
            blurred = gaussian_filter(heatmap_pose, sigma=sig)
            blurred /= np.max(blurred)

            img_small[:, :, 0] += (blurred * 255.)
            img_small[:, :, 2] += (blurred * 255.)

            img_small[np.nonzero(img_small > 255)] = 255

        # Output
        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/visual', im_name),
                          img_small.astype(np.uint8))

    for idx in range(1, 11000 + 1):
        print('{:d}/{:d}'.format(idx, 11000))
        # Read data
        im_name = 'im{:05d}.png'.format(idx)
        img = scipy.misc.imread(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/images', im_name)).astype(
            np.float32)
        heatmap = scipy.misc.imread(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/annot', im_name)).astype(
            np.float32)
        # Parsing
        img_small = scipy.misc.imresize(img, (64, 64), interp='bilinear', mode=None).astype(np.float32)
        for pose_idx in range(0, 16):
            '''
            # Calculate region
            img_height, img_width, _ = img.shape
            long_side = int(np.ceil(scale[idx - 1] * 100.))
            img_center_x, img_center_y = int(center[idx - 1][0]), int(center[idx - 1][1])
            pad_y_min, pad_x_min = img_center_y - long_side, img_center_x - long_side
            x_min, x_max = max(pad_x_min, 0), min(img_center_x + long_side, img_width)
            y_min, y_max = max(pad_y_min, 0), min(img_center_y + long_side, img_height)
            margin_y_min, margin_x = y_min - pad_y_min, x_min - pad_x_min
            # create heatmap
            cord_x, cord_y = int(part[idx-1][pose_idx][0]), int(part[idx-1][pose_idx][1])
            if pose_idx == 6 or pose_idx == 7 or cord_x < 0 or cord_y < 0:
                continue
            cord_y = max((cord_y - pad_y_min), 0) / float(long_side) * 32.
            cord_x = max((cord_x - pad_x_min), 0) / float(long_side) * 32.
            cord_y = min(cord_y, 63)
            cord_x = min(cord_x, 63)
            cord_y, cord_x = int(np.round(cord_y)), int(np.round(cord_x))
            heatmap = np.zeros((64, 64), np.float32)
            heatmap[cord_y, cord_x] = 1
            '''

            heatmap_pose = np.zeros((64, 64), np.float32)
            cord = np.nonzero(heatmap == (pose_idx + 1))
            if pose_idx == 6 or pose_idx == 7 or len(cord[0]) == 0:
                continue
            heatmap_pose[cord] = 1
            blurred = gaussian_filter(heatmap_pose, sigma=sig)
            blurred /= np.max(blurred)

            img_small[:, :, 0] += (blurred * 255.)
            img_small[:, :, 2] += (blurred * 255.)

            img_small[np.nonzero(img_small > 255)] = 255

        # Output
        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/visual', im_name),
                          img_small.astype(np.uint8))


def read_crop_lsp_test():
    """
    """
    sig = 3
    # train images
    for idx in range(1001, 2000 + 1):
        print('{:d}/{:d}'.format(idx, 1000))
        # Read data
        im_name = 'im{:04d}.png'.format(idx)
        img = scipy.misc.imread(os.path.join('/data/vllab1/pose-hg-train/data/LSP/test/images', im_name)).astype(
            np.float32)
        heatmap = scipy.misc.imread(os.path.join('/data/vllab1/pose-hg-train/data/LSP/test/annot', im_name)).astype(
            np.float32)
        # Parsing
        img_small = scipy.misc.imresize(img, (64, 64), interp='bilinear', mode=None).astype(np.float32)
        for pose_idx in range(0, 16):
            '''
            # Calculate region
            img_height, img_width, _ = img.shape
            long_side = int(np.ceil(scale[idx - 1] * 100.))
            img_center_x, img_center_y = int(center[idx - 1][0]), int(center[idx - 1][1])
            pad_y_min, pad_x_min = img_center_y - long_side, img_center_x - long_side
            x_min, x_max = max(pad_x_min, 0), min(img_center_x + long_side, img_width)
            y_min, y_max = max(pad_y_min, 0), min(img_center_y + long_side, img_height)
            margin_y_min, margin_x = y_min - pad_y_min, x_min - pad_x_min
            # create heatmap
            cord_x, cord_y = int(part[idx-1][pose_idx][0]), int(part[idx-1][pose_idx][1])
            if pose_idx == 6 or pose_idx == 7 or cord_x < 0 or cord_y < 0:
                continue
            cord_y = max((cord_y - pad_y_min), 0) / float(long_side) * 32.
            cord_x = max((cord_x - pad_x_min), 0) / float(long_side) * 32.
            cord_y = min(cord_y, 63)
            cord_x = min(cord_x, 63)
            cord_y, cord_x = int(np.round(cord_y)), int(np.round(cord_x))
            heatmap = np.zeros((64, 64), np.float32)
            heatmap[cord_y, cord_x] = 1
            '''

            heatmap_pose = np.zeros((64, 64), np.float32)
            cord = np.nonzero(heatmap == (pose_idx + 1))
            if pose_idx == 6 or pose_idx == 7 or len(cord[0]) == 0:
                continue
            heatmap_pose[cord] = 1
            blurred = gaussian_filter(heatmap_pose, sigma=sig)
            blurred /= np.max(blurred)

            img_small[:, :, 0] += (blurred * 255.)
            img_small[:, :, 2] += (blurred * 255.)

            img_small[np.nonzero(img_small > 255)] = 255

        # Output
        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/LSP/test/visual', im_name),
                          img_small.astype(np.uint8))


def crop_mpii(data_set_dir):
    """
    """
    # All metedata
    filename = '/data/vllab1/pose-hg-train/data/lsp_mpii.h5'
    f = h5py.File(filename, 'r')
    part = f['part']
    center = f['center']
    scale = f['scale']
    imgname = f['imgname']
    istrain = f['istrain']
    visible = f['visible']
    person = f['person']
    # train images
    for idx in range(12000, 40883):
        print('{:d}/{:d}'.format(idx, 40883))
        img_orb_name = imgname[idx]
        im_name = ''
        for char in img_orb_name:
            im_name += chr(int(char))
        im_name = im_name.split('.')[0]
        img = scipy.misc.imread(os.path.join(data_set_dir, im_name + '.jpg')).astype(np.float32)
        # Calculate region
        img_height, img_width, _ = img.shape
        long_side = int(np.ceil(scale[idx] * 100.))
        img_center_x, img_center_y = int(center[idx][0]), int(center[idx][1])
        pad_y_min, pad_x_min = img_center_y - long_side, img_center_x - long_side
        x_min, x_max = max(pad_x_min, 0), min(img_center_x + long_side, img_width)
        y_min, y_max = max(pad_y_min, 0), min(img_center_y + long_side, img_height)
        margin_y_min, margin_x = y_min - pad_y_min, x_min - pad_x_min
        # transplant image
        img_center = img[y_min:y_max, x_min:x_max, :]
        img_center_height, img_center_width, _ = img_center.shape
        img_pad = np.zeros((long_side*2, long_side*2, 3), dtype=np.float32)
        img_pad[margin_y_min:margin_y_min+img_center_height, margin_x:margin_x+img_center_width, :] = img_center
        img_pad = scipy.misc.imresize(img_pad, (256, 256), interp='bilinear', mode=None)
        # create heatmap
        heatmap = np.ones((64, 64), np.uint8) * 255
        for pose_idx in range(0, 16):
            cord_x, cord_y = int(part[idx][pose_idx][0]), int(part[idx][pose_idx][1])
            if visible[idx][pose_idx] == 0:
                continue
            cord_y = max((cord_y - pad_y_min), 0) / float(long_side) * 32.
            cord_x = max((cord_x - pad_x_min), 0) / float(long_side) * 32.
            cord_y = min(cord_y, 63)
            cord_x = min(cord_x, 63)
            cord_y, cord_x = int(np.round(cord_y)), int(np.round(cord_x))
            heatmap[cord_y, cord_x] = (pose_idx + 1)

        # Output
        fname = im_name + '_' + str(person[idx]) + '.png'
        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/mpii/test/images', fname),
                          img_pad.astype(np.float32), format='png')
        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/mpii/test/annot', fname),
                          heatmap.astype(np.uint8), format='png')
        #if idx > 12000 + 10:
        #    break


def read_crop_mpii():
    """
    """
    # All metedata
    filename = '/data/vllab1/pose-hg-train/data/lsp_mpii.h5'
    f = h5py.File(filename, 'r')
    part = f['part']
    center = f['center']
    scale = f['scale']
    imgname = f['imgname']
    istrain = f['istrain']
    visible = f['visible']
    person = f['person']
    sig = 3
    # train images
    for idx in range(12000, 40883):
        print('{:d}/{:d}'.format(idx, 40883))
        '''
        img_orb_name = imgname[idx]
        im_name = ''
        for char in img_orb_name:
            im_name += chr(int(char))
        '''
        img_orb_name = imgname[idx]
        im_name = ''
        for char in img_orb_name:
            im_name += chr(int(char))
        im_name = im_name.split('.')[0] + '_' + str(person[idx])
        img = scipy.misc.imread(os.path.join('/data/vllab1/pose-hg-train/data/mpii/test/images', im_name + '.png')).astype(np.float32)
        heatmap = scipy.misc.imread(os.path.join('/data/vllab1/pose-hg-train/data/mpii/test/annot', im_name) + '.png').astype(np.float32)
        # Parsing
        img_small = scipy.misc.imresize(img, (64, 64), interp='bilinear', mode=None).astype(np.float32)
        for pose_idx in range(0, 16):
            heatmap_pose = np.zeros((64, 64), np.float32)
            cord = np.nonzero(heatmap == (pose_idx + 1))
            #if pose_idx == 6 or pose_idx == 7 or len(cord[0]) == 0:
            #    continue
            if len(cord[0]) == 0:
                continue
            heatmap_pose[cord] = 1
            blurred = gaussian_filter(heatmap_pose, sigma=sig)
            blurred /= np.max(blurred)

            img_small[:, :, 0] += (blurred * 255.)
            img_small[:, :, 2] += (blurred * 255.)

            img_small[np.nonzero(img_small > 255)] = 255

        # Output
        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/mpii/test/visual', im_name + '.png'), img_small.astype(np.uint8))

        #if idx > 12000 + 10:
        #    break


def LSP_heatmap_like_images():
    data_set_dir = '/data/vllab1/pose-hg-train/data/LSP/train/images'
    data = glob(os.path.join(data_set_dir, "*.png"))
    sig = 3

    for index, filePath in enumerate(data):
        print ('%d/%d' % (index, len(data)))
        img_name = filePath.split('/')[-1]
        #img = scipy.misc.imread(filePath).astype(np.float32)
        heatmap = scipy.misc.imread(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/annot', img_name)).astype(np.float32)
        heatmap_like = np.zeros((64, 64, 3)).astype(np.float32)

        for pose_idx in range(0, 3):
            heatmap_pose = np.zeros((64, 64), np.float32)
            cord = np.nonzero(heatmap == (pose_idx + 1))
            #if pose_idx == 6 or pose_idx == 7 or len(cord[0]) == 0:
            #    continue
            if len(cord[0]) == 0:
                continue
            heatmap_pose[cord] = 1
            blurred = gaussian_filter(heatmap_pose, sigma=sig)
            blurred /= np.max(blurred)

            heatmap_like[:, :, pose_idx] += (blurred * 255.)

        heatmap_like[np.nonzero(heatmap_like > 255)] = 255

        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/heatmap_like', img_name), heatmap_like.astype(np.uint8))


def LSP_heatmap_all():
    data_set_dir = '/data/vllab1/pose-hg-train/data/LSP/train/images'
    data = glob(os.path.join(data_set_dir, "*.png"))
    part_array = [0, 1, 3, 4, 10, 11, 13, 14, 8]
    sig = 15
    sig_part = 20
    for index, filePath in enumerate(data):
        print ('%d/%d' % (index, len(data)))
        img_name = filePath.split('/')[-1]
        img = scipy.misc.imread(filePath).astype(np.float32)
        heatmap = scipy.misc.imread(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/annot', img_name)).astype(np.float32)

        heatmap_joints = np.zeros((256, 256, 16)).astype(np.float32)
        for pose_idx in range(0, 16):
            heatmap_joint = np.zeros((256, 256), np.float32)
            cord_y, cord_x = np.nonzero(heatmap == (pose_idx + 1))
            #if pose_idx == 6 or pose_idx == 7 or len(cord[0]) == 0:
            #    continue
            if len(cord_y) == 0:
                continue
            heatmap_joint[cord_y*4, cord_x*4] = 1
            blurred = gaussian_filter(heatmap_joint, sigma=sig)
            blurred /= np.max(blurred)

            heatmap_joints[:, :, pose_idx] += (blurred * 255.)

        heatmap_parts = np.zeros((256, 256, 9)).astype(np.float32)
        for idx, pose_idx in enumerate(part_array):
            heatmap_joint = np.zeros((256, 256), np.float32)
            cord_y, cord_x = np.nonzero(heatmap == (pose_idx + 1))
            cord_y_end, cord_x_end = np.nonzero(heatmap == (pose_idx + 2))
            #if pose_idx == 6 or pose_idx == 7 or len(cord[0]) == 0:
            #    continue
            if len(cord_y) == 0 or len(cord_y_end) == 0:
                continue
            rr, cc = line(cord_y*4, cord_x*4, cord_y_end*4, cord_x_end*4)
            heatmap_joint[rr, cc] = 1
            blurred = gaussian_filter(heatmap_joint, sigma=sig_part)
            blurred /= np.max(blurred)

            heatmap_parts[:, :, idx] += (blurred * 255.)

        img_name = img_name.split('.')[0]
        img_visual = np.copy(img)
        for pose_idx in range(0, 16):
            joint_image = np.copy(img)
            joint_image[:, :, 0] += heatmap_joints[:, :, pose_idx]
            img_visual[:, :, 0] += heatmap_joints[:, :, pose_idx]
            joint_image[np.nonzero(joint_image > 255)] = 255
            scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/heatmap_part',
                                           '{}_{:02d}.png'.format(img_name, pose_idx)), joint_image.astype(np.uint8))
        for pose_idx in range(0, 9):
            joint_image = np.copy(img)
            joint_image[:, :, 2] += heatmap_parts[:, :, pose_idx]
            img_visual[:, :, 2] += heatmap_parts[:, :, pose_idx]
            joint_image[np.nonzero(joint_image > 255)] = 255
            scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/heatmap_part',
                                           '{}_{:02d}_part.png'.format(img_name, part_array[pose_idx])), joint_image.astype(np.uint8))

        img_visual[np.nonzero(img_visual > 255)] = 255
        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/heatmap_part',
                                       '{}_visual.png'.format(img_name)), img_visual.astype(np.uint8))

        if index > 10:
            break


def mpii_heatmap_all():
    data_set_dir = '/data/vllab1/pose-hg-train/data/mpii/train/images'
    data = glob(os.path.join(data_set_dir, "*.png"))
    part_array = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14]
    sig = 15
    sig_part = 20
    for index, filePath in enumerate(data):
        print ('%d/%d' % (index, len(data)))
        img_name = filePath.split('/')[-1]
        img = scipy.misc.imread(filePath).astype(np.float32)
        heatmap = scipy.misc.imread(os.path.join('/data/vllab1/pose-hg-train/data/mpii/train/annot', img_name)).astype(np.float32)

        heatmap_joints = np.zeros((256, 256, 16)).astype(np.float32)
        for pose_idx in range(0, 16):
            heatmap_joint = np.zeros((256, 256), np.float32)
            cord_y, cord_x = np.nonzero(heatmap == (pose_idx + 1))
            #if pose_idx == 6 or pose_idx == 7 or len(cord[0]) == 0:
            #    continue
            if len(cord_y) == 0:
                continue
            heatmap_joint[cord_y*4, cord_x*4] = 1
            blurred = gaussian_filter(heatmap_joint, sigma=sig)
            blurred /= np.max(blurred)

            heatmap_joints[:, :, pose_idx] += (blurred * 255.)

        heatmap_parts = np.zeros((256, 256, 13)).astype(np.float32)
        for idx, pose_idx in enumerate(part_array):
            heatmap_joint = np.zeros((256, 256), np.float32)
            cord_y, cord_x = np.nonzero(heatmap == (pose_idx + 1))
            cord_y_end, cord_x_end = np.nonzero(heatmap == (pose_idx + 2))
            #if pose_idx == 6 or pose_idx == 7 or len(cord[0]) == 0:
            #    continue
            if len(cord_y) == 0 or len(cord_y_end) == 0:
                continue
            rr, cc = line(cord_y*4, cord_x*4, cord_y_end*4, cord_x_end*4)
            heatmap_joint[rr, cc] = 1
            blurred = gaussian_filter(heatmap_joint, sigma=sig_part)
            blurred /= np.max(blurred)

            heatmap_parts[:, :, idx] += (blurred * 255.)

        img_name = img_name.split('.')[0]
        img_visual = np.copy(img)
        for pose_idx in range(0, 16):
            joint_image = np.copy(img)
            joint_image[:, :, 0] += heatmap_joints[:, :, pose_idx]
            img_visual[:, :, 0] += heatmap_joints[:, :, pose_idx]
            joint_image[np.nonzero(joint_image > 255)] = 255
            scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/mpii/train/heatmap_part',
                                           '{}_{:02d}.png'.format(img_name, pose_idx)), joint_image.astype(np.uint8))
        for pose_idx in range(0, 13):
            joint_image = np.copy(img)
            joint_image[:, :, 2] += heatmap_parts[:, :, pose_idx]
            img_visual[:, :, 2] += heatmap_parts[:, :, pose_idx]
            joint_image[np.nonzero(joint_image > 255)] = 255
            scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/mpii/train/heatmap_part',
                                           '{}_{:02d}_part.png'.format(img_name, part_array[pose_idx])), joint_image.astype(np.uint8))

        img_visual[np.nonzero(img_visual > 255)] = 255
        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/mpii/train/heatmap_part',
                                       '{}_visual.png'.format(img_name)), img_visual.astype(np.uint8))

        if index > 10:
            break


if __name__ == '__main__':
    #crop_images('/data/vllab1/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/val')
    crop_images('/data/vllab1/dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/val')

