from __future__ import division
import time
from glob import glob
from six.moves import xrange

from ops import *
from utils import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DCGAN_city(object):
    def __init__(self, sess, config, z_dim=128,  gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, y_dim=None):
        """
        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.crop = config.crop

        self.batch_size = config.batch_size
        self.sample_num = config.sample_num

        self.input_height = config.input_height
        self.input_width = config.input_width
        self.output_height = config.output_height
        self.output_width = config.output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # 128, 256
        self.s_h, self.s_w = self.output_height, self.output_width
        # 64, 128
        self.s_h2, self.s_w2 = conv_out_size_same(self.s_h, 2), conv_out_size_same(self.s_w, 2)
        # 32, 64
        self.s_h4, self.s_w4 = conv_out_size_same(self.s_h2, 2), conv_out_size_same(self.s_w2, 2)
        # 16, 32
        self.s_h8, self.s_w8 = conv_out_size_same(self.s_h4, 2), conv_out_size_same(self.s_w4, 2)
        # 8, 16
        self.s_h16, self.s_w16 = conv_out_size_same(self.s_h8, 2), conv_out_size_same(self.s_w8, 2)

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.num_of_class = config.num_of_class

        self.dataset_name = config.dataset_name
        self.image_dir = config.image_dir
        self.input_fname_pattern = config.input_fname_pattern
        self.checkpoint_dir = config.checkpoint_dir

        self.data = glob(os.path.join(self.image_dir, self.input_fname_pattern))
        np.random.shuffle(self.data)
        self.c_dim = imread(self.data[0]).shape[-1]
        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        keep_probability = tf.placeholder(tf.float32, name="keep_probability")
        image = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width, self.c_dim], name="input_image")
        annotation = tf.placeholder(tf.float32,shape=[None, self.output_height, self.output_width, self.num_of_class], name="annotation")
        with tf.variable_scope("fcn"):
            pred_annotation, logits = fcn.inference(image, keep_probability)
            loss_fcn = tf.losses.mean_squared_error(labels=annotation, predictions=logits)
        fcn_variable = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='fcn')

        image_dims = [self.output_height, self.output_width, self.c_dim]

        self.inputs = tf.placeholder(
          tf.float32, [None] + image_dims, name='real_images')
        self.z = tf.placeholder(
          tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.inputs)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=2)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # s_h2, s_w2
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # s_h4, s_w4
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # s_h8, s_w8
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # s_h16, s_w16
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            # reshape
            batch_size = tf.shape(image)[0]
            h3_reshape = tf.reshape(h3, [batch_size, self.s_h16*self.s_w16*self.df_dim*8])
            # fully
            h4 = linear(h3_reshape, self.s_h16*self.s_w16*self.df_dim*8, 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, self.z_dim, self.gf_dim*8*self.s_h16*self.s_w16, 'g_h0_lin', with_w=True)
            # s_h16, s_w16
            self.h0 = tf.reshape(
                self.z_, [-1, self.s_h16, self.s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))
            # s_h8, s_w8
            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [-1, self.s_h8, self.s_w8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))
            # s_w4, s_w4
            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [-1, self.s_h4, self.s_w4, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))
            # s_h2, s_w2
            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [-1, self.s_h2, self.s_w2, self.gf_dim*1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))
            # s_h, s_w
            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [-1, self.s_h, self.s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def train(self, config):
        self.d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                  .minimize(self.g_loss, var_list=self.g_vars)
        tf.global_variables_initializer().run(session=self.sess)

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
          self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs_{}".format(config.dataset_name), self.sess.graph)

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        counter = 1
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        self.train_update(config, counter)

    def train_update(self, config, counter=1):
        sample_files = self.data[0:self.sample_num]
        sample = [simple_get_image(sample_file) for sample_file in sample_files]
        sample_inputs = np.array(sample).astype(np.float32)
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        start_time = time.time()

        for epoch in xrange(config.epoch):
            np.random.shuffle(self.data)
            batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [simple_get_image(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([self.d_optim, self.d_sum],
                  feed_dict={ self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([self.g_optim, self.g_sum],
                  feed_dict={ self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([self.g_optim, self.g_sum],
                  feed_dict={ self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                with self.sess.as_default():
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                    errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                  % (epoch, idx, batch_idxs,
                    time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    samples_G, d_loss, g_loss = self.sess.run(
                      [self.G, self.d_loss, self.g_loss],
                      feed_dict={
                          self.z: sample_z,
                          self.inputs: sample_inputs,
                      },
                    )
                    manifold_h = int(np.ceil(np.sqrt(samples_G.shape[0])))
                    manifold_w = int(np.floor(np.sqrt(samples_G.shape[0])))
                    save_images(samples_G, [manifold_h, manifold_w],
                          './{}/train_{:06d}.png'.format(config.sample_dir, counter))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def test(self, config):
        tf.global_variables_initializer().run(session=self.sess)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
            sample_idxs = len(self.data) // config.sample_num

            for idx in xrange(0, sample_idxs):
                print('{:d}/{:d}'.format(idx, sample_idxs))
                sample_z = np.random.uniform(-1, 1, [config.sample_num, self.z_dim]).astype(np.float32)
                samples_G = self.sess.run(self.G, feed_dict={self.z: sample_z})
                manifold_h = int(np.ceil(np.sqrt(samples_G.shape[0])))
                manifold_w = int(np.floor(np.sqrt(samples_G.shape[0])))
                save_images(samples_G, [manifold_h, manifold_w],
                            './{}/test_{:06d}.png'.format(config.test_dir, idx))
        else:
            print(" [!] Load failed...")
            raise Exception("[!] Train a model first, then run test mode")

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)



    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
