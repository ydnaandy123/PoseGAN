"""
Adopt DCGAN to train lsp
"""
import numpy as np
from model_fcn import FCN
import os
from utils import pp, show_all_variables, config_check
import utils
from glob import glob
from scipy.ndimage.filters import gaussian_filter
import scipy.misc
import time
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("dataset_name", "fcn_pose", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("mode", "train", "network mode [train, test]")

flags.DEFINE_string("image_dir", "/data/vllab1/pose-hg-train/data/mpii/train/images", "The directory of images")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")

flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("test_dir", "test", "Directory name to save the image samples [samples]")

flags.DEFINE_integer("sample_num", 8, "The number of sample images [64]")
flags.DEFINE_integer("batch_size", 8, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 256, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 256, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 256, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 256, "The size of the output images to produce. If None, same value as output_height [None]")

flags.DEFINE_integer("epoch", 10000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

# FCN
flags.DEFINE_integer("num_of_class", 16, "Number of class [25]")
flags.DEFINE_string("vgg_dir", "../../checkpoint/", "Path to vgg model mat")
flags.DEFINE_float("learning_rate_fcn", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1_fcn", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_boolean("debug", False, "True for debugging, False for nothing [False]")
FLAGS = flags.FLAGS


def load(checkpoint_dir, model_dir, saver, sess):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def save(checkpoint_dir, step, model_dir, saver, sess):
    model_name = "FCN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)


def main(_):
    # Configuration
    pp.pprint(flags.FLAGS.__flags)
    config, model_dir = config_check(FLAGS)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    """
    # FCN
    # image should be in range 0-255
    # will be subtracted by mean(vgg) during inference
    """
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")
    image = tf.placeholder(tf.float32, shape=[None, config.input_height, config.input_width, 3], name="input_image")
    annotation = tf.placeholder(tf.float32, shape=[None, config.output_height, config.output_width, config.num_of_class], name="annotation")
    with tf.variable_scope("fcn"):
        fcn = FCN(flags=config, num_of_class=config.num_of_class)
        pred_annotation, logits = fcn.inference(image, keep_probability)
        #pred_heatmap = tf.nn.tanh(logits)
        pred_heatmap = logits
        loss_fcn = tf.losses.mean_squared_error(labels=annotation, predictions=pred_heatmap)
    fcn_variable = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='fcn')
    """
    # Loss
    """
    fcn_optim = tf.train.AdamOptimizer(config.learning_rate_fcn, beta1=config.beta1_fcn) \
        .minimize(loss_fcn, var_list=fcn_variable)
    """
    " Launch session
    """
    show_all_variables()
    saver = tf.train.Saver(max_to_keep=2)
    sess = tf.Session(config=run_config)
    sess.run(tf.global_variables_initializer())
    """
    " Load or create model
    """
    could_load, checkpoint_counter = load(config.checkpoint_dir, model_dir, saver, sess)
    counter = 0
    if could_load:
        counter = checkpoint_counter
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    if FLAGS.mode == "train":
        sig = 15
        sig_part = 20
        data = glob(os.path.join(config.image_dir, config.input_fname_pattern))
        np.random.shuffle(data)

        sample_files = data[0:config.sample_num]
        sample = [scipy.misc.imread(sample_file) for sample_file in sample_files]
        sample_inputs = np.array(sample).astype(np.float32)

        start_time = time.time()
        for epoch in xrange(config.epoch):
            np.random.shuffle(data)
            batch_idxs = min(len(data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch, annote = [], []
                for batch_file in batch_files:
                    batch.append(scipy.misc.imread(batch_file))
                    img_name = batch_file.split('/')[-1]
                    heatmap = scipy.misc.imread(
                        os.path.join('/data/vllab1/pose-hg-train/data/mpii/train/annot', img_name)).astype(np.float32)
                    heatmap_joints = np.zeros((256, 256, 16)).astype(np.float32)
                    for pose_idx in range(0, 16):
                        heatmap_joint = np.zeros((256, 256), np.float32)
                        cord_y, cord_x = np.nonzero(heatmap == (pose_idx + 1))
                        # if pose_idx == 6 or pose_idx == 7 or len(cord[0]) == 0:
                        #    continue
                        if len(cord_y) == 0:
                            continue
                        heatmap_joint[cord_y * 4, cord_x * 4] = 1
                        blurred = gaussian_filter(heatmap_joint, sigma=sig)
                        blurred /= np.max(blurred)

                        heatmap_joints[:, :, pose_idx] += (blurred * 255.)
                    annote.append(heatmap_joints)

                batch_images = np.array(batch).astype(np.float32)
                batch_annotes = np.array(annote).astype(np.float32) / 127.5 - 1

                feed_dict_fcn = {image: batch_images, annotation: batch_annotes, keep_probability: 0.85}
                sess.run(fcn_optim, feed_dict=feed_dict_fcn)

                if np.mod(counter, 5) == 0:
                    err_fcn = sess.run(loss_fcn, feed_dict={image: batch_images, annotation: batch_annotes, keep_probability: 1.0})
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, idx, batch_idxs,
                             time.time() - start_time, err_fcn, err_fcn))

                if np.mod(counter, 10) == 0:
                    prediction = sess.run(pred_heatmap, feed_dict={image: batch_images, keep_probability: 1.0})
                    manifold_h = int(np.ceil(np.sqrt(prediction.shape[0])))
                    manifold_w = int(np.floor(np.sqrt(prediction.shape[0])))
                    utils.save_images(prediction[:, :, :, :3], [manifold_h, manifold_w],
                          './{}/train_{:06d}_pred.png'.format(config.sample_dir, counter))
                    utils.save_images(batch_images, [manifold_h, manifold_w],
                          './{}/train_{:06d}_image.png'.format(config.sample_dir, counter))
                    utils.save_images((batch_annotes[:, :, :, :3] + 1.) * 127.5, [manifold_h, manifold_w],
                          './{}/train_{:06d}_gt.png'.format(config.sample_dir, counter))

                if np.mod(counter, 500) == 2:
                    save(config.checkpoint_dir, counter, model_dir, saver, sess)

                counter += 1

    elif FLAGS.mode == 'test':
        pass
        #dcgan.test(config)
    else:
        if not load(FLAGS.checkpoint_dir, model_dir, saver, sess)[0]:
            raise Exception("[!] Train a model first, then run test mode")

if __name__ == '__main__':
    tf.app.run()
