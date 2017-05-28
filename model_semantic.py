from __future__ import division
import time
from glob import glob
import re
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dropout, Activation
from keras.regularizers import l2

from ops import *
from utils import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class SEMANTIC(object):
    def __init__(self, sess, config, z_dim=100, gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024):
        """
        Args:
          sess: TensorFlow session
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
        """
        # Experiments setting
        self.sess = sess
        self.name = config.name
        self.need_condition = config.need_condition
        self.need_g1 = config.need_g1
        self.g1_mode = config.g1_mode
        self.debug = config.debug
        # Network
        self.batch_size = config.batch_size
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        # TODO: generalize
        self.num_of_class = 34
        # hyper parameter
        self.lambda_gp = config.lambda_gp
        self.lambda_g1 = config.lambda_g1
        # Directory
        self.checkpoint_dir = config.checkpoint_dir
        self.sample_dir = config.sample_dir
        self.vgg_dir = config.vgg_dir
        # Input image
        self.dataset_name = config.dataset_name
        self.image_dir = config.image_dir
        self.condition_dir = config.condition_dir
        self.input_fname_pattern = config.input_fname_pattern
        # Image format
        self.data = glob(os.path.join(self.image_dir, self.input_fname_pattern))
        np.random.shuffle(self.data)
        self.image_height = config.image_height
        self.image_width = config.image_width
        self.image_dim = config.image_dim
        self.condition_height = config.condition_height
        self.condition_width = config.condition_width
        self.condition_dim = config.condition_dim
        # Default setting
        self.c_dim = config.image_dim
        self.c_heatmap = (config.name.find('heatmap') != -1)
        self.c_heatmap_3 = (config.name.find('heatmap_3') != -1)
        self.c_heatmap_all = (config.name.find('heatmap_all') != -1)
        self.c_heatmap_compress = (config.name.find('heatmap_compress') != -1)
        self.c_label = (config.name.find('label') != -1)
        # Sample
        self.sample_num = config.sample_num
        self.manifold_h_sample = int(np.ceil(np.sqrt(self.sample_num)))
        self.manifold_w_sample = int(np.floor(np.sqrt(self.sample_num)))
        self.manifold_h_batch = int(np.ceil(np.sqrt(self.batch_size)))
        self.manifold_w_batch = int(np.floor(np.sqrt(self.batch_size)))
        self.sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
        self.sample_files = self.data[0:self.sample_num]
        self.sample_images, self.sample_conditions = self.data_sample()
        # Subsample factor
        self.image_sh, self.image_sw, self.image_subsample_factor = [self.image_height], [self.image_width], 0
        self.condition_sh, self.condition_sw, self.condition_subsample_factor = \
            [self.condition_height], [self.condition_width], 0
        self.subsample()
        # Model Graph
        self.is_training, self.inputs, self.conditions, self.z = None, None, None, None
        self.G, self.D_logits_real, self.D_logits_fake = None, None, None
        self.d_loss_real, self.d_loss_fake, self.g_loss, self.d_loss = None, None, None, None
        self.g_vars, self.d_vars, self.extra_update_ops, self.saver = None, None, None, None
        # Training phase
        self.d_optim, self.g_optim, self.summary_op, self.writer = None, None, None, None
        self.build_model()

    def subsample(self):
        # subsample factor of image
        min_s = min(self.image_height, self.image_width)
        while min_s >= 2:
            s_h, s_w = conv_out_size_same(
                self.image_sh[self.image_subsample_factor], 2), \
                       conv_out_size_same(self.image_sw[self.image_subsample_factor], 2)
            self.image_sh.append(s_h)
            self.image_sw.append(s_w)
            min_s = min(s_h, s_w)
            self.image_subsample_factor += 1
        min_s = min(self.condition_height, self.condition_width)
        # subsample factor of condition
        while min_s >= 2:
            s_h, s_w = conv_out_size_same(
                self.condition_sh[self.condition_subsample_factor], 2), \
                       conv_out_size_same(self.condition_sw[self.condition_subsample_factor], 2)
            self.condition_sh.append(s_h)
            self.condition_sw.append(s_w)
            min_s = min(s_h, s_w)
            self.condition_subsample_factor += 1

    def get_batch(self, files):
        """
        :param files: list of directories of input images
        :return: mini-batch of (images, con.itions) pair, normalized in -1.0 ~ 1.0 (float32)
        """
        if self.c_heatmap_3:
            images, conditions = get_image_condition_pose_mpii_big(files, self.condition_dir, channel_num=3)
        elif self.c_heatmap_all:
            images, conditions = get_image_condition_pose_mpii_big(files, self.condition_dir)
        elif self.c_heatmap_compress:
            images, conditions = get_image_condition_pose_mpii_big(files, self.condition_dir)
            images = norm_image(heatmap_visual_mpii(denorm_image(images)))
        else:
            images, conditions = get_image_condition(files, self.condition_dir)
        return images, conditions

    def data_sample(self):
        """
        1.sample files
        2.de-normalized and merge sample files for visualize
        :return: mini-batch of (images, con.itions) pair, normalized in -1.0 ~ 1.0 (float32)
        """
        if self.c_heatmap_3:
            sample_images, sample_conditions = get_image_condition_pose_mpii_big(
                self.sample_files, self.condition_dir, channel_num=3)
            images_visual = merge(denorm_image(sample_images), [self.manifold_h_sample, self.manifold_w_sample])
        elif self.c_heatmap_all:
            sample_images, sample_conditions = get_image_condition_pose_mpii_big(self.sample_files, self.condition_dir)
            images_visual = merge(heatmap_visual_mpii(denorm_image(sample_images)),
                                  [self.manifold_h_sample, self.manifold_w_sample])
        elif self.c_heatmap_compress:
            sample_images, sample_conditions = get_image_condition_pose_mpii_big(self.sample_files, self.condition_dir)
            images_visual = merge(heatmap_visual_mpii(denorm_image(sample_images)),
                                  [self.manifold_h_sample, self.manifold_w_sample])
            sample_images = norm_image(heatmap_visual_mpii(denorm_image(sample_images)))
        else:
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

    def data_visual(self, data):
        """
        1. de-normalized from [-1.0, 1.0] to [0, 255]
        :param data:
        :return: visualized data in range (0, 255)
        """
        if self.c_heatmap_all:
            images_visual = merge(heatmap_visual_mpii(denorm_image(data)),
                                  [self.manifold_h_sample, self.manifold_w_sample])
        else:
            images_visual = merge(denorm_image(data), [self.manifold_h_sample, self.manifold_w_sample])
        return images_visual

    def loss_wgan_gp(self):
        # Get real_data, fake_data, disc_fake, disc_real
        batch_size = tf.shape(self.inputs)[0]
        image_all_dim = self.image_height * self.image_width * self.image_dim
        real_data = tf.reshape(self.inputs, [batch_size, image_all_dim])
        fake_data = tf.reshape(self.G, [batch_size, image_all_dim])
        disc_fake = self.D_logits_fake
        disc_real = self.D_logits_real
        # Standard WGAN loss
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
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
        # G1
        if self.need_g1:
            if self.g1_mode == 'L1':
                gen_cost += self.lambda_g1 * tf.reduce_mean(tf.abs(differences))

        return gen_cost, disc_cost

    def build_model(self):
        # Input place holder
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.inputs = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_dim],
                                     name='real_images')
        self.conditions = tf.placeholder(tf.float32,
                                         [None, self.condition_height, self.condition_width, self.condition_dim],
                                         name='conditions')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        # Generator and discriminator
        with tf.variable_scope("GEN"):
            # activate with tanh
            self.G = self.generator(self.z, self.conditions, training=self.is_training)
        with tf.variable_scope("DIS"):
            # activate with tanh, none
            self.D_logits_real = self.discriminator(self.inputs, self.conditions,
                                                    reuse=False, training=self.is_training)
            self.D_logits_fake = self.discriminator(self.G, self.conditions, reuse=True, training=self.is_training)
        # Loss
        self.g_loss, self.d_loss = self.loss_wgan_gp()
        # Variables and saver
        self.g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='GEN')
        self.d_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='DIS')
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.saver = tf.train.Saver(max_to_keep=2)

    def vgg_net(self, weights, image):
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )

        net = {}
        current = image
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                bias = get_variable(bias.reshape(-1), name=name + "_b")
                current = conv2d_basic(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
                if self.debug:
                    add_activation_summary(current)
            elif kind == 'pool':
                current = avg_pool_2x2(current)
            net[name] = current

        return net

    def inference(self, image, keep_prob):
        """
        Semantic segmentation network definition
        :param image: input image. Should have values in range 0-255
        :param keep_prob:
        :return:
        """
        print("setting up vgg initialized conv layers ...")
        model_data = scipy.io.loadmat(os.path.join(self.vgg_dir, 'imagenet-vgg-verydeep-19.mat'))
        mean = model_data['normalization'][0][0][0]
        mean_pixel = np.mean(mean, axis=(0, 1))
        weights = np.squeeze(model_data['layers'])

        # TODO: need to normalized in -1~1 ?
        processed_image = image - mean_pixel

        with tf.variable_scope("inference"):
            image_net = self.vgg_net(weights, processed_image)
            conv_final_layer = image_net["conv5_3"]

            pool5 = max_pool_2x2(conv_final_layer)

            w6 = tf.get_variable('w6', [7, 7, 512, 4096], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b6 = tf.get_variable('b6', [4096], initializer=tf.constant_initializer(0.0))
            conv6 = conv2d_basic(pool5, w6, b6)
            relu6 = tf.nn.relu(conv6, name="relu6")
            if self.debug:
                add_activation_summary(relu6)
            relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

            w7 = tf.get_variable('w7', [1, 1, 4096, 4096], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b7 = tf.get_variable('b7', [4096], initializer=tf.constant_initializer(0.0))
            conv7 = conv2d_basic(relu_dropout6, w7, b7)
            relu7 = tf.nn.relu(conv7, name="relu7")
            if self.debug:
                add_activation_summary(relu7)
            relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

            w8 = tf.get_variable('w8', [1, 1, 4096, self.num_of_class],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            b8 = tf.get_variable('b8', [self.num_of_class], initializer=tf.constant_initializer(0.0))
            conv8 = conv2d_basic(relu_dropout7, w8, b8)
            # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

            # now to upscale to actual image size
            # TODO: no relu?
            deconv_shape1 = image_net["pool4"].get_shape().as_list()
            conv_t1 = deconv2d(tf.nn.relu(conv8), [-1, deconv_shape1[1], deconv_shape1[2], deconv_shape1[3]],
                               k_h=4, k_w=4, name='conv_t1')
            fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
            # TODO: better writting
            deconv_shape2 = image_net["pool3"].get_shape()
            conv_t2 = deconv2d(tf.nn.relu(fuse_1), deconv_shape2,
                               k_h=4, k_w=4, name='conv_t2')
            fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")
            # TODO: better writting
            shape = tf.shape(image)
            conv_t3 = deconv2d(tf.nn.relu(fuse_2), deconv_shape3,
                               k_h=16, k_w=16, d_h=8, d_w=8, name='conv_t3')

            annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

        return tf.expand_dims(annotation_pred, dim=3), conv_t3

    def discriminator(self, image, condition, reuse=False, training=True):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            if self.need_condition:
                # [concat, conv, lrelu]
                image = tf.concat([image, condition], 3)
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                # [concat, conv, bn, lrelu]
                condition_s1 = max_pool_2x2(condition)
                h0 = tf.concat([h0, condition_s1], 3)
                h1 = lrelu(bn(conv2d(h0, self.df_dim * 2, name='d_h1_conv'), training=training))

                condition_s2 = max_pool_2x2(condition_s1)
                h1 = tf.concat([h1, condition_s2], 3)
                h2 = lrelu(bn(conv2d(h1, self.df_dim * 4, name='d_h2_conv'), training=training))

                condition_s3 = max_pool_2x2(condition_s2)
                h2 = tf.concat([h2, condition_s3], 3)
                h3 = lrelu(bn(conv2d(h2, self.df_dim * 8, name='d_h3_conv'), training=training))
                # reshape to linear (s_16 = 2^4)
                # condition_s4 = max_pool_2x2(condition_s3)
                # h3 = tf.concat([h3, condition_s4], 3)
                h3_reshape = tf.reshape(h3, [tf.shape(image)[0], self.image_sh[4] * self.image_sw[4] * self.df_dim * 8])
                # [fully, bn, lrelu]
                # h4 = lrelu(bn(linear(h3_reshape, self.dfc_dim, 'd_h3_lin'), training=training))
                # fully
                h4 = linear(h3_reshape, 1, 'd_h4_lin')

                return h4
            else:
                # [Conv, lrelu]
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                # [conv, bn, lrelu] x 3
                h1 = lrelu(bn(conv2d(h0, self.df_dim * 2, name='d_h1_conv'), training=training))
                h2 = lrelu(bn(conv2d(h1, self.df_dim * 4, name='d_h2_conv'), training=training))
                h3 = lrelu(bn(conv2d(h2, self.df_dim * 8, name='d_h3_conv'), training=training))
                # reshape to linear (s_16 = 2^4)
                h3_reshape = tf.reshape(h3, [tf.shape(image)[0], self.image_sh[4] * self.image_sw[4] * self.df_dim * 8])
                # fully
                h4 = linear(h3_reshape, 1, 'd_h3_lin')

                return h4

    def dense_block(self, layers_count, filters, previous_layer, model_layers, level):
        model_layers[level] = {}
        for i in range(layers_count):
            model_layers[level]['b_norm'+str(i+1)] = BatchNormalization(
                axis=3, gamma_regularizer=l2(0.0001), beta_regularizer=l2(0.0001))(previous_layer)
            model_layers[level]['act'+str(i+1)] = Activation('relu')(model_layers[level]['b_norm'+str(i+1)])
            model_layers[level]['conv'+str(i+1)] = Conv2D(
                filters,   kernel_size=(3, 3), padding='same', kernel_initializer="he_uniform",
                data_format='channels_last')(model_layers[level]['act'+str(i+1)])
            model_layers[level]['drop_out'+str(i+1)] = Dropout(0.2)(model_layers[level]['conv'+str(i+1)])
            previous_layer  = model_layers[level]['drop_out'+str(i+1)]
        # print(model_layers)
        return model_layers[level]['drop_out'+ str(layers_count)] # return last layer of this level

    def generator(self, z, condition, training=True):
        with tf.variable_scope("generator"):
            if self.need_condition:
                first_conv = Conv2D(48, kernel_size=(3, 3), padding='same',
                                    kernel_initializer="he_uniform",
                                    kernel_regularizer=l2(0.0001),
                                    data_format='channels_last')(condition)
                enc_model_layers = {}
                layer_1_down = self.dense_block(5, 108, first_conv, enc_model_layers,
                                               'layer_1_down')  # 5*12 = 60 + 48 = 108

                # TODO: need of z?
                # image is (256 x 256 x input_c_dim)
                e1 = conv2d(condition, self.gf_dim, name='g_e1_conv')
                # e1 is (128 x 128 x self.gf_dim)
                e2 = bn(conv2d(lrelu(e1), self.gf_dim * 2, name='g_e2_conv'), training=training)
                # e2 is (64 x 64 x self.gf_dim*2)
                e3 = bn(conv2d(lrelu(e2), self.gf_dim * 4, name='g_e3_conv'), training=training)
                # e3 is (32 x 32 x self.gf_dim*4)
                e4 = bn(conv2d(lrelu(e3), self.gf_dim * 8, name='g_e4_conv'), training=training)
                # e4 is (16 x 16 x self.gf_dim*8)
                e5 = bn(conv2d(lrelu(e4), self.gf_dim * 8, name='g_e5_conv'), training=training)
                # e5 is (8 x 8 x self.gf_dim*8)
                e6 = bn(conv2d(lrelu(e5), self.gf_dim * 8, name='g_e6_conv'), training=training)
                # e6 is (4 x 4 x self.gf_dim*8)
                e7 = bn(conv2d(lrelu(e6), self.gf_dim * 8, name='g_e7_conv'), training=training)
                # e7 is (2 x 2 x self.gf_dim*8)
                e8 = conv2d(lrelu(e7), self.gf_dim * 8, name='g_e8_conv')
                # e8 is (1 x 1 x self.gf_dim*8)

                d1 = bn(deconv2d(tf.nn.relu(e8), [-1, self.image_sh[7], self.image_sw[7], self.gf_dim * 8],
                                 name='g_d1'), training=training)
                d1 = tf.concat([d1, e7], 3)
                # d1 is (2 x 2 x self.gf_dim*8*2)
                d2 = bn(deconv2d(tf.nn.relu(d1), [-1, self.image_sh[6], self.image_sw[6], self.gf_dim * 8],
                                 name='g_d2'), training=training)
                d2 = tf.concat([d2, e6], 3)
                # d2 is (4 x 4 x self.gf_dim*8*2)
                d3 = bn(deconv2d(tf.nn.relu(d2), [-1, self.image_sh[5], self.image_sw[5], self.gf_dim * 8],
                                 name='g_d3'), training=training)
                d3 = tf.concat([d3, e5], 3)
                # d3 is (8 x 8 x self.gf_dim*8*2)
                d4 = bn(deconv2d(tf.nn.relu(d3), [-1, self.image_sh[4], self.image_sw[4], self.gf_dim * 8],
                                 name='g_d4'), training=training)
                d4 = tf.concat([d4, e4], 3)
                # d4 is (16 x 16 x self.gf_dim*8*2)
                d5 = bn(deconv2d(tf.nn.relu(d4), [-1, self.image_sh[3], self.image_sw[3], self.gf_dim * 4],
                                 name='g_d5'), training=training)
                d5 = tf.concat([d5, e3], 3)
                # d5 is (32 x 32 x self.gf_dim*4*2)
                d6 = bn(deconv2d(tf.nn.relu(d5), [-1, self.image_sh[2], self.image_sw[2], self.gf_dim * 2],
                                 name='g_d6'), training=training)
                d6 = tf.concat([d6, e2], 3)
                # d6 is (64 x 64 x self.gf_dim*2*2)
                d7 = bn(deconv2d(tf.nn.relu(d6), [-1, self.image_sh[1], self.image_sw[1], self.gf_dim],
                                 name='g_d7'), training=training)
                d7 = tf.concat([d7, e1], 3)
                # d7 is (128 x 128 x self.gf_dim*1*2)
                d8 = deconv2d(tf.nn.relu(d7), [-1, self.image_sh[0], self.image_sw[0], self.image_dim], name='g_d8')
                # d8 is (256 x 256 x output_c_dim)
                return tf.nn.tanh(d8)
            else:
                # [fully, bn, relu] to (s_14 * s_16 * gf_dim * 8)
                h0 = linear(z, self.image_sh[4] * self.image_sw[4] * self.gf_dim * 8, 'g_h0_lin')
                h0 = tf.nn.relu(bn(tf.reshape(
                    h0, [tf.shape(z)[0], self.image_sh[4], self.image_sw[4], self.gf_dim * 8]), training=training))
                # [deconv, bn, relu] x 3
                d1 = tf.nn.relu(bn(deconv2d(
                    h0, [-1, self.image_sh[3], self.image_sw[3], self.gf_dim * 4], name='g_d1'), training=training))
                d2 = tf.nn.relu(bn(deconv2d(
                    d1, [-1, self.image_sh[2], self.image_sw[2], self.gf_dim * 2], name='g_d2'), training=training))
                d3 = tf.nn.relu(bn(deconv2d(
                    d2, [-1, self.image_sh[1], self.image_sw[1], self.gf_dim * 1], name='g_d3'), training=training))
                # deconv
                d4 = deconv2d(d3, [-1, self.image_sh[0], self.image_sw[0], self.image_dim], name='g_d4')
                return tf.nn.tanh(d4)

    def train(self, config):
        # Training optimizer
        with tf.control_dependencies(self.extra_update_ops):
            self.d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, beta2=config.beta2) \
                .minimize(self.d_loss, var_list=self.d_vars)
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

    # TODO: BN is_training?
    def train_update(self, config, counter=0):
        start_time = time.time()
        sample_feed = {self.is_training: True, self.z: self.sample_z,
                       self.inputs: self.sample_images, self.conditions: self.sample_conditions}
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
                batch_feed = {self.is_training: True, self.z: batch_z,
                              self.inputs: batch_images, self.conditions: batch_conditions}
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
                    images_visual = self.data_visual(samples_g)
                    scipy.misc.imsave('./{}/train_{:06d}.png'.format(self.sample_dir, counter),
                                      images_visual.astype(np.uint8))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                # Checkpoint, save model
                if np.mod(counter, 500) == 100:
                    print('Model saved...')
                    self.save(config.checkpoint_dir, counter)

                counter += 1

    def test_z(self, config):
        tf.global_variables_initializer().run(session=self.sess)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
            if not os.path.exists(config.test_dir):
                os.makedirs(config.test_dir)
            sample_idxs = min(len(self.data) // config.sample_num, 5)

            for idx in range(0, sample_idxs):
                print('{:d}/{:d}'.format(idx, sample_idxs))
                if self.need_condition:
                    sample_z = np.random.uniform(-1, 1, [config.sample_num, self.z_dim]).astype(np.float32)
                    batch_files = self.data[idx * config.sample_num:(idx + 1) * config.sample_num]
                else:
                    sample_z = np.random.uniform(-1, 1, [config.sample_num, self.z_dim]).astype(np.float32)
                    samples_G = self.sess.run(self.G, feed_dict={self.z: sample_z})
        else:
            print(" [!] Load failed...")
            raise Exception("[!] Train a model first, then run test mode")

    def test_condition(self, config):
        tf.global_variables_initializer().run(session=self.sess)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
            if not os.path.exists(config.test_dir):
                os.makedirs(config.test_dir)
            sample_idxs = len(self.data) // config.sample_num

            for idx in range(0, sample_idxs):
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
        return "{}_{}_{}_{}".format(self.name, self.batch_size, self.image_height, self.image_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
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
