from __future__ import division
import time
from glob import glob
import scipy.io
import re

from ops import *
from utils import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class Model(object):
    def __init__(self, sess, config, need_flip=True, z_dim=100, gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024):
        # Experiments setting
        self.sess = sess
        self.name = config.name
        self.mode = config.mode
        self.sample_num = config.sample_num
        self.batch_size = config.batch_size
        self.train_size = config.train_size
        self.debug = config.debug
        self.need_flip = need_flip
        # Network training feed
        self.dataset_name = config.dataset_name
        self.image_dir = config.image_dir
        self.condition_dir = config.condition_dir
        self.input_fname_pattern = config.input_fname_pattern
        self.image_height = config.image_height
        self.image_width = config.image_width
        self.image_dim = config.image_dim
        self.condition_height = config.condition_height
        self.condition_width = config.condition_width
        self.condition_dim = config.condition_dim
        # Network sample(testing) feed
        self.image_dir_val = config.image_dir_val
        self.condition_dir_val = config.condition_dir_val
        self.manifold_h_sample = int(np.ceil(np.sqrt(self.sample_num)))
        self.manifold_w_sample = int(np.floor(np.sqrt(self.sample_num)))
        # Calling virtual method
        self.data = self.get_training_data()
        np.random.shuffle(self.data)
        self.data_val = self.get_valid_data()
        np.random.shuffle(self.data_val)
        self.sample_files = self.data_val[0:self.sample_num]
        # Directory
        self.checkpoint_dir = config.checkpoint_dir
        self.test_dir = config.test_dir
        self.sample_dir = config.sample_dir

    def get_training_data(self):
        pass

    def get_valid_data(self):
        pass

    def get_sample(self):
        """
        1.sample files
        2.de-normalized and merge sample files for visualize
        :return: mini-batch of (images, conditions) pair, normalized in -1.0 ~ 1.0 (float32)
        """
        sample_images, sample_conditions = get_image_condition(self.sample_files, self.condition_dir)
        images_visual = merge(denorm_image(sample_images), [self.manifold_h_sample, self.manifold_w_sample])
        print('DEAULT IMAGE TYPE: RGB-3')

        scipy.misc.imsave('./{}/sample_0_images.png'.format(self.sample_dir), images_visual.astype(np.uint8))
        conditions_visual = merge(denorm_image(sample_conditions), [self.manifold_h_sample, self.manifold_w_sample])
        scipy.misc.imsave('./{}/sample_2_conditions.png'.format(self.sample_dir), conditions_visual.astype(np.uint8))

        images_visual = np.array(images_visual * 0.5 + conditions_visual * 0.5).astype(np.float32)
        images_visual[np.nonzero(images_visual > 255.)] = 255.
        scipy.misc.imsave('./{}/sample_1_image_condition.png'.format(self.sample_dir),
                          images_visual.astype(np.uint8))

        return sample_images, sample_conditions

    def get_batch(self, files):
        """
        :param files: list of directories of input images
        :return: mini-batch of (images, con.itions) pair, normalized in -1.0 ~ 1.0 (float32)
        """
        images, conditions = get_image_condition(files, self.condition_dir)
        return images, conditions

    def output_visual(self, data):
        """
        1. de-normalized from [-1.0, 1.0] to [0, 255]
        :param data:
        :return: visualized data in range (0, 255)
        """
        images_visual = merge(denorm_image(data), [self.manifold_h_sample, self.manifold_w_sample])
        return images_visual

    def loss_wgan_gp_base(self):
        disc_fake = self.D_logits_fake
        disc_real = self.D_logits_real
        # Standard WGAN loss
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        # Gradient penalty
        # Get real_data, fake_data, disc_fake, disc_real
        batch_size = tf.shape(self.images)[0]
        image_all_dim = self.image_height * self.image_width * self.image_dim
        real_data = tf.reshape(self.images, [batch_size, image_all_dim])
        fake_data = tf.reshape(self.G, [batch_size, image_all_dim])
        # Gradient penalty
        alpha = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)
        differences = fake_data - real_data
        interpolates = real_data + (alpha * differences)
        interpolates = tf.reshape(interpolates, [batch_size, self.image_height, self.image_width, self.image_dim])
        with tf.variable_scope("DIS"):
            # TODO: why [interpolates]?
            d_logits_ = self.discriminator(interpolates, self.conditions, reuse=True, training=self.is_training)
            gradients = tf.gradients(d_logits_, [interpolates])[0]
        gradients = tf.reshape(gradients, [batch_size, image_all_dim])
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        disc_cost += self.lambda_gp * gradient_penalty

        return gen_cost, disc_cost

    def build_model(self):
        # Input place holder
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.images = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_dim],
                                     name='real_images')
        self.conditions = tf.placeholder(tf.float32,
                                         [None, self.condition_height, self.condition_width, self.condition_dim],
                                         name='conditions')
        self.keep_prob = tf.placeholder(tf.float32, name="keep_probability")
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        # Network_architect

        # Variables and saver
        self.g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='GEN')
        self.d_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='DIS')
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.saver = tf.train.Saver(max_to_keep=2)

    def train(self):
        # Training optimizer
        with tf.control_dependencies(self.extra_update_ops):
            self.d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, beta2=config.beta2) \
                .minimize(self.d_loss, var_list=self.d_vars)
            self.d_optim = tf.constant(0.)
            self.g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, beta2=config.beta2) \
                .minimize(self.g_loss, var_list=self.g_vars)
        # Training summary logs
        if self.debug:
            self.summary_op = tf.summary.merge([
                tf.summary.histogram("histogram/z", self.z),
                tf.summary.histogram("histogram/D_real", self.D_logits_real),
                tf.summary.histogram("histogram/D_fake", self.D_logits_fake),
                tf.summary.histogram("histogram/G", self.G),

                tf.summary.scalar("loss/d_loss", self.d_loss),
                tf.summary.scalar("loss/g_loss", self.g_loss),
            ])
            self.writer = tf.summary.FileWriter("./{}_logs".format(config.name), self.sess.graph)
        # Initialize and restore model
        tf.global_variables_initializer().run(session=self.sess)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            counter = 0
            print(" [!] Load failed...")
        self.train_update(config, counter)

    def train_update(self, counter=0):
        start_time = time.time()
        sample_feed = {self.is_training: False, self.z: self.sample_z, self.keep_prob: 1.0,
                       self.images: self.sample_images, self.conditions: self.sample_conditions}
        # every epoch
        for epoch in range(config.epoch):
            np.random.shuffle(self.data)
            batch_idxs = min(len(self.data), config.train_size) // config.batch_size
            # every mini-batch
            for idx in range(0, batch_idxs):
                # Get feeds
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)
                batch_files = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_images, batch_conditions = self.get_batch(batch_files)
                batch_feed = {self.is_training: True, self.z: batch_z, self.keep_prob: 0.8,
                              self.images: batch_images, self.conditions: batch_conditions}
                # Update network
                _ = self.sess.run(self.d_optim, feed_dict=batch_feed)
                _ = self.sess.run(self.g_optim, feed_dict=batch_feed)
                _ = self.sess.run(self.g_optim, feed_dict=batch_feed)
                # Training logs
                err_d, err_g = \
                    self.sess.run([self.d_loss, self.g_loss], feed_dict=batch_feed)
                print("Epoch: [%2d] [%4d/%4d] [%7d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" %
                      (epoch, idx, batch_idxs, counter, time.time() - start_time, err_d, err_g))
                if np.mod(counter, 5) == 0:
                    summary_str = self.sess.run(self.summary_op, feed_dict=batch_feed)
                    self.writer.add_summary(summary_str, counter)
                # Sample logs and visualized images
                if np.mod(counter, 100) == 0:
                    samples_g, d_loss, g_loss = \
                        self.sess.run([self.G, self.d_loss, self.g_loss], feed_dict=sample_feed)
                    images_visual = self.output_visual(samples_g)
                    scipy.misc.imsave('./{}/train_{:06d}.png'.format(self.sample_dir, counter),
                                      images_visual.astype(np.uint8))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                # Checkpoint, save model
                if np.mod(counter, 500) == 100:
                    print('Model saved...')
                    self.save(config.checkpoint_dir, counter)

                counter += 1

    def pred(self):
        tf.global_variables_initializer().run(session=self.sess)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
            result_dir = './results_test'
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            # TODO: full image size
            pred_dir = '/data/vllab1/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/test'
            # pred_dir ='/data/vllab1/Github/pose_gan/dataset/CITYSCAPES_DATASET/valid/image'

            pred_image = []
            for folder in os.listdir(pred_dir):
                path = os.path.join(pred_dir, folder, "*.png")
                pred_image.extend(glob(path))
            # pred_image = glob(os.path.join(pred_dir, self.input_fname_pattern))

            # TODO: test data batch?
            for idx in range(0, len(pred_image)):
                # Get feeds
                batch_files = pred_image[idx]
                name = batch_files.split('/')[-1]
                print('{:d}/{:d}: {}'.format(idx, len(pred_image), name))
                # TODO: silly way to expand dims
                #batch_image = [simple_get_image(batch_files)]
                batch_image = [scipy.misc.imresize(scipy.misc.imread(batch_files), 0.125).astype(np.float32) / 127.5 - 1.]
                sample_feed = {self.is_training: False, self.keep_prob: 1.0, self.conditions: batch_image}
                samples_g = self.sess.run(self.G, feed_dict=sample_feed)

                samples_g = np.argmax(np.squeeze(samples_g, axis=0), axis=2)
                samples_g_out = np.zeros((1024, 2048), dtype=np.uint8)
                for m in range(1024):
                    for n in range(2048):
                        cur_m, cur_n = int(np.floor(m/8)), int(np.floor(n/8))
                        samples_g_out[m, n] = samples_g[cur_m, cur_n]
                scipy.misc.imsave('./{}/{}'.format(result_dir, name), samples_g_out.astype(np.uint8))
                label_v = label_id_visual_(samples_g_out)
                scipy.misc.imsave('./{}/{}'.format('visual_test', name), label_v.astype(np.uint8))
                #break
        else:
            print(" [!] Load failed...")
            raise Exception("[!] Train a model first, then run test mode")

    @property
    def model_dir(self):
        return "{}_{}".format(self.name, self.batch_size)

    def save(self, checkpoint_dir, step):
        model_name = "model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

