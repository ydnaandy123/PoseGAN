from glob import glob
import numpy as np
import os
import scipy.misc
import h5py
from scipy.ndimage.filters import gaussian_filter


def crop_images(dataset_dir):
    """
    Read all images under the different folders
    Crop, resize and store them
    example code:
        crop_images(CITYSCAPES_dir)
    """
    data = []
    for folder in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, folder, "*.png")
        data.extend(glob(path))

    for index, filePath in enumerate(data):
        print ('{}/{}'.format(index, len(data)))

        img = scipy.misc.imread(filePath).astype(np.uint8)
        img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
        scipy.misc.imsave('/data/vllab1/dataset/CITYSCAPES/coarse_resize/' + filePath.split('/')[-1], img)
        #break


def crop_images_same_dir(data_set_dir):
    """
    Read all images under the same folder
    Crop, resize and store them
    """
    data = glob(os.path.join(data_set_dir, "*.png"))

    for index, filePath in enumerate(data):
        print ('%d/%d' % (index, len(data)))

        img = scipy.misc.imread(filePath).astype(np.float32)
        #img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
        #scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_random/' + filePath.split('/')[-1],
        #                  img[offs_h[index]:offs_h_end[index], offs_w[index]:offs_w_end[index] :])
        scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom_192/' + filePath.split('/')[-1],
                          img[0:192, :, :])
        #break


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




if __name__ == '__main__':
    #crop_lsp('/data/vllab1/pose-hg-train/data/LSP/images')
    read_crop_lsp()

