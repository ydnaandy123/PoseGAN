from __future__ import division
import time
from glob import glob
from six.moves import xrange

from ops import *
from utils import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DCGAN_conditional_G1(object):
    def __init__(self, sess, config, z_dim=100,  gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, needCondition=False):
        """
        Args:
          sess: TensorFlow session
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
        """
        self.sess = sess
        self.needCondition = needCondition
        self.crop = config.crop

        self.L1_lambda = config.L1_lambda
        self.batch_size = config.batch_size
        self.sample_num = config.sample_num

        self.image_height = config.image_height
        self.image_width = config.image_width
        self.image_dim = config.image_dim
        self.condition_height = config.condition_height
        self.condition_width = config.condition_width
        self.condition_dim = config.condition_dim

        self.dataset_name = config.dataset_name
        self.image_dir = config.image_dir
        self.condition_dir = config.condition_dir
        self.input_fname_pattern = config.input_fname_pattern
        self.checkpoint_dir = config.checkpoint_dir

        self.data = glob(os.path.join(self.image_dir, self.input_fname_pattern))
        np.random.shuffle(self.data)
        self.c_dim = config.image_dim
        self.grayscale = (self.c_dim == 1)

        # 256, 256
        self.s_h, self.s_w = self.condition_height, self.condition_width
        # 128, 128
        self.s_h2, self.s_w2 = conv_out_size_same(self.s_h, 2), conv_out_size_same(self.s_w, 2)
        # 64, 64
        self.s_h4, self.s_w4 = conv_out_size_same(self.s_h2, 2), conv_out_size_same(self.s_w2, 2)
        # 32, 32
        self.s_h8, self.s_w8 = conv_out_size_same(self.s_h4, 2), conv_out_size_same(self.s_w4, 2)
        # 16, 16
        self.s_h16, self.s_w16 = conv_out_size_same(self.s_h8, 2), conv_out_size_same(self.s_w8, 2)
        # 8, 8
        self.s_h32, self.s_w32 = conv_out_size_same(self.s_h16, 2), conv_out_size_same(self.s_w16, 2)
        # 4, 4
        self.s_h64, self.s_w64 = conv_out_size_same(self.s_h32, 2), conv_out_size_same(self.s_w32, 2)
        # 2, 2
        self.s_h128, self.s_w128 = conv_out_size_same(self.s_h64, 2), conv_out_size_same(self.s_w64, 2)

        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        # self.g_bn_d6 = batch_norm(name='g_bn_d6')
        # self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.build_model()

    def build_model(self):

        self.inputs = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_dim], name='real_images')
        self.conditions = tf.placeholder(tf.float32, [None, self.condition_height, self.condition_width, self.condition_dim], name='conditions')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        with tf.variable_scope("GEN"):
            self.G = self.generator(self.z, self.conditions)
        with tf.variable_scope("DIS"):
            self.D, self.D_logits = self.discriminator(self.inputs, self.conditions)
            self.D_, self.D_logits_ = self.discriminator(self.G, self.conditions, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = histogram_summary("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        # TODO: G1 loss
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) + \
                      self.L1_lambda * tf.reduce_mean(tf.abs(self.inputs - self.G))
        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        #t_vars = tf.trainable_variables()
        #self.d_vars = [var for var in t_vars if 'd_' in var.name]
        #self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='GEN')
        self.d_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='DIS')

        self.saver = tf.train.Saver(max_to_keep=2)

    def discriminator(self, image, condition, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            if self.needCondition:
                # concat image with condition
                # TODO generalize
                condition = max_pool_4x4(condition)
                x = tf.concat([image, condition], 3)
                # TODO: why the first two layer no need to bN?
                # conv and concat (64x64)
                h = lrelu(conv2d(x, self.c_dim + self.condition_dim, d_h=1, d_w=1, name='d_h_conv'))
                h = tf.concat([h, condition], 3)
                # s_h2, s_w2  (64x64)
                h0 = lrelu(conv2d(h, self.df_dim + self.condition_dim, name='d_h0_conv'))
                condition_s2 = max_pool_2x2(condition)
                h0 = tf.concat([h0, condition_s2], 3)
                # s_h4, s_w4  (32x32)
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2 + self.condition_dim, name='d_h1_conv')))
                condition_s4 = max_pool_2x2(condition_s2)
                h1 = tf.concat([h1, condition_s4], 3)
                # s_h8, s_w8  (16x16)
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4 + self.condition_dim, name='d_h2_conv')))
                condition_s8 = max_pool_2x2(condition_s4)
                h2 = tf.concat([h2, condition_s8], 3)
                # s_h16, s_w16  (8x8)
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8 + self.condition_dim, name='d_h3_conv')))
                condition_s16 = max_pool_2x2(condition_s8)
                h3 = tf.concat([h3, condition_s16], 3)
                # TODO generalize, not use (s_h64, s_w64)
                # reshape
                batch_size = tf.shape(image)[0]
                feature_dim = self.s_h64*self.s_w64*(self.df_dim*8 + self.condition_dim*2)
                h3_reshape = tf.reshape(h3, [batch_size, feature_dim])
                condition_s16_reshape = tf.reshape(condition_s16, [batch_size, self.s_h64*self.s_w64*self.condition_dim])
                # fully
                h4 = lrelu(self.d_bn4(linear(h3_reshape, feature_dim, self.dfc_dim, 'd_h4_lin')))
                h4 = tf.concat([h4, condition_s16_reshape], 1)
                # fully
                h5 = linear(h4, self.dfc_dim + self.s_h64*self.s_w64*self.condition_dim, 1, 'd_h5_lin')

                return tf.nn.sigmoid(h5), h5
            else:
                # TODO: why the first two layer no need to bN?
                # conv and concat (64x64)
                h = lrelu(conv2d(image, self.c_dim, d_h=1, d_w=1, name='d_h_conv'))
                # s_h2, s_w2  (64x64)
                h0 = lrelu(conv2d(h, self.df_dim, name='d_h0_conv'))
                # s_h4, s_w4  (32x32)
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                # s_h8, s_w8  (16x16)
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                # s_h16, s_w16  (8x8)
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
                # TODO generalize, not use (s_h64, s_w64)
                # reshape
                batch_size = tf.shape(image)[0]
                feature_dim = self.s_h64 * self.s_w64 * self.df_dim * 8
                h3_reshape = tf.reshape(h3, [batch_size, feature_dim])
                # fully
                h4 = lrelu(self.d_bn4(linear(h3_reshape, feature_dim, self.dfc_dim, 'd_h4_lin')))
                # fully
                h5 = linear(h4, self.dfc_dim, 1, 'd_h5_lin')

                return tf.nn.sigmoid(h5), h5

    def generator(self, z, condition):
        with tf.variable_scope("generator") as scope:
            if self.needCondition:
                # TODO why the first conv no need to BN?
                # condition is (256 x 256 x condition_dim)
                # e1 is (128 x 128 x self.gf_dim)
                e1 = lrelu(conv2d(condition, self.gf_dim, name='g_e1_conv'))
                # e2 is (64 x 64 x self.gf_dim*2)
                e2 = lrelu(self.g_bn_e2(conv2d(e1, self.gf_dim*2, name='g_e2_conv')))
                # e3 is (32 x 32 x self.gf_dim*4)
                e3 = lrelu(self.g_bn_e3(conv2d(e2, self.gf_dim*4, name='g_e3_conv')))
                # e4 is (16 x 16 x self.gf_dim*8)
                e4 = lrelu(self.g_bn_e4(conv2d(e3, self.gf_dim*8, name='g_e4_conv')))
                # e5 is (8 x 8 x self.gf_dim*8)
                e5 = lrelu(self.g_bn_e5(conv2d(e4, self.gf_dim*8, name='g_e5_conv')))
                # e6 is (4 x 4 x self.gf_dim*8)
                e6 = lrelu(self.g_bn_e6(conv2d(e5, self.gf_dim*8, name='g_e6_conv')))
                # e7 is (2 x 2 x self.gf_dim*8)
                e7 = lrelu(self.g_bn_e7(conv2d(e6, self.gf_dim*8, name='g_e7_conv')))
                # e8 is (1 x 1 x self.gf_dim*8)
                e8 = lrelu(self.g_bn_e8(conv2d(e7, self.gf_dim*8, name='g_e8_conv')))

                z = tf.concat([z, tf.squeeze(e8)], 1)
                batch_size = tf.shape(z)[0]

                # fully to (1 x gfc_dim)
                h0 = tf.nn.relu(self.g_bn0(linear(z, self.z_dim + self.gf_dim*8, self.gfc_dim, 'g_h0_lin')))
                h0 = tf.concat([h0, tf.squeeze(e8)], 1)
                # fully to (2 x 2 x self.gf_dim*8)
                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gfc_dim + self.gf_dim*8, self.s_h128 * self.s_w128 * self.gf_dim * 8, 'g_h1_lin')))
                h1 = tf.reshape(h1, [batch_size, self.s_h128, self.s_w128, self.gf_dim * 8])
                h1 = tf.concat([h1, e7], 3)
                # deconv to (4 x 4 x self.gf_dim*8)
                d2 = tf.nn.relu(self.g_bn_d2(deconv2d(h1, [-1, self.s_h64, self.s_w64, self.gf_dim * 8], name='g_d2')))
                d2 = tf.concat([d2, e6], 3)
                # deconv to (8 x 8 x self.gf_dim*4)
                d3 = tf.nn.relu(self.g_bn_d3(deconv2d(d2, [-1, self.s_h32, self.s_w32, self.gf_dim * 4], name='g_d3')))
                d3 = tf.concat([d3, e5], 3)
                # d4 is (16 x 16 x self.gf_dim*2)
                d4 = tf.nn.relu(self.g_bn_d4(deconv2d(d3, [-1, self.s_h16, self.s_w16, self.gf_dim * 2], name='g_d4')))
                d4 = tf.concat([d4, e4], 3)
                # d5 is (32 x 32 x self.gf_dim*1)
                d5 = tf.nn.relu(self.g_bn_d5(deconv2d(d4, [-1, self.s_h8, self.s_w8, self.gf_dim * 1], name='g_d5')))
                d5 = tf.concat([d5, e3], 3)
                # TODO why the last layer no need to BN?
                # d6 is (64 x 64 x self.image_dim)
                d6 = deconv2d(d5, [-1, self.s_h4, self.s_w4, self.image_dim], name='g_d6')
                # TODO sigmoid? tanh?
                return tf.nn.tanh(d6)
                #return tf.nn.sigmoid(d6)
            else:
                batch_size = tf.shape(z)[0]
                # fully to (1 x gfc_dim)
                h0 = tf.nn.relu(self.g_bn0(linear(z, self.z_dim, self.gfc_dim, 'g_h0_lin')))
                # fully to (2 x 2 x self.gf_dim*8)
                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gfc_dim, self.s_h128 * self.s_w128 * self.gf_dim * 8, 'g_h1_lin')))
                h1 = tf.reshape(h1, [batch_size, self.s_h128, self.s_w128, self.gf_dim * 8])
                # deconv to (4 x 4 x self.gf_dim*8)
                d2 = tf.nn.relu(self.g_bn_d2(deconv2d(h1, [-1, self.s_h64, self.s_w64, self.gf_dim * 8], name='g_d2')))
                # deconv to (8 x 8 x self.gf_dim*4)
                d3 = tf.nn.relu(self.g_bn_d3(deconv2d(d2, [-1, self.s_h32, self.s_w32, self.gf_dim * 4], name='g_d3')))
                # d4 is (16 x 16 x self.gf_dim*2)
                d4 = tf.nn.relu(self.g_bn_d4(deconv2d(d3, [-1, self.s_h16, self.s_w16, self.gf_dim * 2], name='g_d4')))
                # d5 is (32 x 32 x self.gf_dim*1)
                d5 = tf.nn.relu(self.g_bn_d5(deconv2d(d4, [-1, self.s_h8, self.s_w8, self.gf_dim * 1], name='g_d5')))
                # TODO why the last layer no need to BN?
                # d6 is (64 x 64 x self.image_dim)
                d6 = deconv2d(d5, [-1, self.s_h4, self.s_w4, self.image_dim], name='g_d6')
                # TODO sigmoid? tanh?
                return tf.nn.tanh(d6)
                #return tf.nn.sigmoid(d6)

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
        counter = 0
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        self.train_update(config, counter)

    def train_update(self, config, counter=0):
        sample_files = self.data[0:self.sample_num]
        sample_images, sample_conditions = get_image_condition_pose(sample_files, config.condition_dir)
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
        manifold_h = int(np.ceil(np.sqrt(sample_images.shape[0])))
        manifold_w = int(np.floor(np.sqrt(sample_images.shape[0])))

        conditions_visual = (merge(sample_conditions, [manifold_h, manifold_w]) + 1.) * 127.5
        images_visual = merge(heatmap_visual(sample_images), [manifold_h, manifold_w])
        scipy.misc.imsave('./{}/sample_0_images.png'.format(config.sample_dir), images_visual.astype(np.uint8))
        scipy.misc.imsave('./{}/sample_1_conditions.png'.format(config.sample_dir), conditions_visual.astype(np.uint8))
        images_visual_big = scipy.misc.imresize(images_visual, 4.) + conditions_visual
        images_visual_big[np.nonzero(images_visual_big > 255.)] = 255.
        scipy.misc.imsave('./{}/sample_2_image_condition.png'.format(config.sample_dir),
                          images_visual_big.astype(np.uint8))

        start_time = time.time()

        for epoch in xrange(config.epoch):
            np.random.shuffle(self.data)
            batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_images, batch_conditions = get_image_condition_pose(batch_files, config.condition_dir)
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([self.d_optim, self.d_sum],
                  feed_dict={ self.inputs: batch_images, self.conditions: batch_conditions, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([self.g_optim, self.g_sum],
                  feed_dict={ self.inputs: batch_images, self.z: batch_z, self.conditions: batch_conditions})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([self.g_optim, self.g_sum],
                  feed_dict={ self.inputs: batch_images, self.z: batch_z, self.conditions: batch_conditions})
                self.writer.add_summary(summary_str, counter)

                with self.sess.as_default():
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.conditions: batch_conditions})
                    errD_real = self.d_loss_real.eval({self.inputs: batch_images, self.conditions: batch_conditions})
                    errG = self.g_loss.eval({self.inputs: batch_images, self.z: batch_z, self.conditions: batch_conditions})

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                  % (epoch, idx, batch_idxs,
                    time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 0:
                    samples_G, d_loss, g_loss = self.sess.run(
                      [self.G, self.d_loss, self.g_loss],
                      feed_dict={
                          self.z: sample_z,
                          self.inputs: sample_images,
                          self.conditions: sample_conditions
                      },
                    )
                    images_visual = merge(heatmap_visual(samples_G), [manifold_h, manifold_w])

                    scipy.misc.imsave('./{}/train_yo_{:06d}.png'.format(config.sample_dir, counter),
                                      merge((samples_G[:, :, :, :3] + 1.) * 127.5, [manifold_h, manifold_w]).astype(np.uint8))
                    scipy.misc.imsave('./{}/train_{:06d}.png'.format(config.sample_dir, counter),
                                      images_visual.astype(np.uint8))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 20:
                    print('Model saved...')
                    self.save(config.checkpoint_dir, counter)

                counter += 1

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
            self.image_height, self.image_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                os.path.join(checkpoint_dir, model_name),
                global_step=step)

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
