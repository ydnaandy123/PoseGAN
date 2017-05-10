from glob import glob
import numpy as np
import os
import scipy.misc
import h5py

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
    for idx in range(1, 11000 + 1):
        print('{:d}/{:d}'.format(idx, 11000))
        im_name = 'im{:05d}.jpg'.format(idx)
        img = scipy.misc.imread(os.path.join(data_set_dir, im_name)).astype(np.float32)
        img_height, img_width, _ = img.shape
        #img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
        long_side = int(np.ceil(scale[idx - 1] * 100.))

        img_center_x, img_center_y = int(center[idx - 1][0]), int(center[idx - 1][1])
        pad_y_min, pad_x_min = img_center_y - long_side, img_center_x - long_side
        x_min, x_max = max(pad_x_min, 0), min(img_center_x + long_side, img_width)
        y_min, y_max = max(pad_y_min, 0), min(img_center_y + long_side, img_height)
        margin_y_min, margin_x = y_min - pad_y_min, x_min - pad_x_min
        img_center = img[y_min:y_max, x_min:x_max, :]
        img_center_height, img_center_width, _ = img_center.shape
        img_pad = np.zeros((long_side*2, long_side*2, 3))
        img_pad[margin_y_min:margin_y_min+img_center_height, margin_x:margin_x+img_center_width, :] = img_center

        scipy.misc.imsave(os.path.join('/data/vllab1/pose-hg-train/data/LSP/train/images', im_name), img_pad)

        break


if __name__ == '__main__':
    crop_lsp('/data/vllab1/pose-hg-train/data/LSP/images')

