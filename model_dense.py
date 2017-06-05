from __future__ import division
from model import Model
from glob import glob
import time
from utils_lite import *
from ops import *


class DENSECity(Model):
    def __init__(self, sess, config, gf_dim=64, df_dim=64):
        # init
        self.label_dir = config.label_dir
        self.label_dir_val = config.label_dir_val
        self.num_of_class = config.num_of_class
        super(DENSECity, self).__init__(sess=sess, config=config)
        # Hyper parameter
        self.epoch = config.epoch
        self.learning_rate = config.learning_rate
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.lambda_gp = config.lambda_gp
        self.lambda_loss_generate = config.lambda_loss_generate
        self.lambda_loss_classify = config.lambda_loss_classify
        # Network architect
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        # ===================================================
        # -----------------Model graph-----------------------
        # ===================================================
        # Feed
        self.is_training = self.keep_prob = None
        self.images = self.conditions = self.labels = None
        self.batch_size_realtime = self.image_all_dim = self.image_shape = None
        # GEN, DIS
        self.fake_images = self.fake_labels = None
        self.D_logits_real = self.D_logits_fake = None
        # ===================================================
        # -----------------Loss------------------------------
        # ===================================================
        # Loss Adv
        self.loss_G_adv = self.loss_D_adv = None
        # Loss Generate
        self.loss_generate = None
        # Loss Classify
        self.loss_classify = None
        # Loss All
        self.g_loss = self.d_loss = None
        # Training phase
        self.extra_update_ops = self.saver = None
        self.g_vars = self.d_vars = None
        self.d_optim = self.g_optim = self.summary_op = self.writer = None

        # 128, 256
        self.s_h, self.s_w = self.condition_height, self.condition_width
        # 64, 128
        self.s_h2, self.s_w2 = conv_out_size_same(self.s_h, 2), conv_out_size_same(self.s_w, 2)
        # 32, 64
        self.s_h4, self.s_w4 = conv_out_size_same(self.s_h2, 2), conv_out_size_same(self.s_w2, 2)
        # 16, 32
        self.s_h8, self.s_w8 = conv_out_size_same(self.s_h4, 2), conv_out_size_same(self.s_w4, 2)
        # 8, 16
        self.s_h16, self.s_w16 = conv_out_size_same(self.s_h8, 2), conv_out_size_same(self.s_w8, 2)
        # 4, 8
        self.s_h32, self.s_w32 = conv_out_size_same(self.s_h16, 2), conv_out_size_same(self.s_w16, 2)
        # 2, 4
        self.s_h64, self.s_w64 = conv_out_size_same(self.s_h32, 2), conv_out_size_same(self.s_w32, 2)
        # 1, 2
        self.s_h128, self.s_w128 = conv_out_size_same(self.s_h64, 2), conv_out_size_same(self.s_w64, 2)

    # ===================================================
    # -----------------Files processing------------------
    # ===================================================
    @overrides(Model)
    def get_training_data(self):
        data = sorted(glob(os.path.join(self.image_dir, self.input_fname_pattern)))
        data_condition = sorted(glob(os.path.join(self.condition_dir, self.input_fname_pattern)))
        data_label = sorted(glob(os.path.join(self.label_dir, self.input_fname_pattern)))
        data = zip(data, data_condition, data_label)
        return data

    @overrides(Model)
    def get_valid_data(self):
        data = sorted(glob(os.path.join(self.image_dir_val, self.input_fname_pattern)))
        data_condition = sorted(glob(os.path.join(self.condition_dir_val, self.input_fname_pattern)))
        data_label = sorted(glob(os.path.join(self.label_dir_val, self.input_fname_pattern)))
        data = zip(data, data_condition, data_label)
        return data

    @overrides(Model)
    def get_sample(self):
        # Get sample
        image_files, conditions_files, label_files = zip(*self.sample_files)
        sample_images = get_batch_images_norm(image_files, self.need_flip, scale=1.0)
        sample_conditions = get_batch_images_norm(conditions_files, self.need_flip, scale=1.0)
        sample_labels = get_batch_images(label_files, self.need_flip, scale=1.0, dim=0)
        # Sample visual
        images_visual = merge(denorm_image(sample_images), [self.manifold_h_sample, self.manifold_w_sample])
        scipy.misc.imsave('./{}/sample_0_images.png'.format(self.sample_dir), images_visual.astype(np.uint8))
        conditions_visual = merge(denorm_image(sample_conditions), [self.manifold_h_sample, self.manifold_w_sample])
        scipy.misc.imsave('./{}/sample_2_conditions.png'.format(self.sample_dir), conditions_visual.astype(np.uint8))
        # Images and conditions blending
        images_visual = np.array(images_visual * 0.5 + conditions_visual * 0.5).astype(np.float32)
        images_visual[np.nonzero(images_visual > 255.)] = 255.
        scipy.misc.imsave('./{}/sample_1_image_condition.png'.format(self.sample_dir),
                          images_visual.astype(np.uint8))

        return sample_images, sample_conditions, sample_labels

    @overrides(Model)
    def get_batch(self, files):
        # Get batch
        image_files, conditions_files, label_files = zip(*files)
        images = get_batch_images_norm(image_files, self.need_flip, scale=0.5)
        conditions = get_batch_images_norm(conditions_files, self.need_flip, scale=0.5)
        labels = get_batch_images(label_files, self.need_flip, scale=0.5, dim=0)
        return images, conditions, labels

    @overrides(Model)
    def output_visual(self, data):
        images_visual = merge(denorm_image(data), [self.manifold_h_sample, self.manifold_w_sample])
        return images_visual

    def label_visual(self, data):
        images_visual = merge(label_visual(data), [self.manifold_h_sample, self.manifold_w_sample])
        return images_visual

    # ===================================================
    # -----------------Generator-------------------------
    # ===================================================
    def generator(self, condition, training=True, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            n_filters_first_conv = 48
            n_pool = 4
            n_layers_per_block = 4
            n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
            growth_rate = 12
            #####################
            # First Convolution #
            #####################
            # We perform a first convolution. All the features maps will
            # be stored in the tensor called stack (the Tiramisu)
            stack = conv2d(condition, output_dim=n_filters_first_conv, k_h=3, k_w=3, d_h=1, d_w=1, name='first_conv')
            # The number of feature maps in the stack is stored in the variable n_filters
            n_filters = n_filters_first_conv
            #####################
            # Downsampling path #
            #####################
            skip_connection_list = []
            for i in range(n_pool):
                # Dense Block
                for j in range(n_layers_per_block[i]):
                    # Compute new feature maps
                    name = 'dense_down_{:d}_{:d}'.format(i, j)
                    l = bn_relu_conv(stack, growth_rate, keep_prob=self.keep_prob, training=training, name=name)
                    # And stack it : the Tiramisu is growing
                    stack = tf.concat([stack, l], 3)
                    n_filters += growth_rate
                # At the end of the dense block, the current stack is stored in the skip_connections list
                skip_connection_list.append(stack)
                # Transition Down
                name = 'td_{:d}'.format(i)
                stack = transition_down(stack, n_filters, keep_prob=self.keep_prob, training=training, name=name)
            #####################
            #     Bottleneck    #
            #####################
            # We store now the output of the next dense block in a list.
            # We will only upsample these new feature maps
            block_to_upsample = []
            # Dense Block
            for j in range(n_layers_per_block[n_pool]):
                name = 'dense_bottleneck_{:d}'.format(j)
                l = bn_relu_conv(stack, growth_rate, keep_prob=self.keep_prob, training=training, name=name)
                block_to_upsample.append(l)
                stack = tf.concat([stack, l], 3)
            skip_connection_list = skip_connection_list[::-1]
            #######################
            #   Upsampling path   #
            #######################
            for i in range(n_pool):
                # Transition Up ( Upsampling + concatenation with the skip connection)
                name = 'tu_{:d}'.format(i)
                # TODO: currently same with input feature maps
                n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
                stack = transition_up(skip_connection_list[i], block_to_upsample, n_filters_keep, training=training,
                                      name=name)
                # Dense Block
                block_to_upsample = []
                for j in range(n_layers_per_block[n_pool + i + 1]):
                    name = 'dense_up_{:d}_{:d}'.format(i, j)
                    l = bn_relu_conv(stack, growth_rate, keep_prob=self.keep_prob, training=training, name=name)
                    block_to_upsample.append(l)
                    stack = tf.concat([stack, l], 3)
            #####################
            #      Softmax      #
            #####################
            # TODO: somthing
            # self.output_layer = SoftmaxLayer(stack, n_classes)
            output_layer_gen = conv2d(stack, self.image_dim, d_h=1, d_w=1, name='outputlayer_gen')
            output_layer = conv2d(stack, self.num_of_class, d_h=1, d_w=1, name='outputlayer_class')
            # d8 is (256 x 256 x output_c_dim)
            return tf.nn.tanh(output_layer_gen), output_layer

    def discriminator(self, image, condition, reuse=False, training=True):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            image = tf.concat([image, condition], 3)
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # [conv, bn, lrelu] x 3
            h1 = lrelu(bn(conv2d(h0, self.df_dim * 2, name='d_h1_conv'), training=training))

            h2 = lrelu(bn(conv2d(h1, self.df_dim * 4, name='d_h2_conv'), training=training))

            h3 = lrelu(bn(conv2d(h2, self.df_dim * 8, name='d_h3_conv'), training=training))

            h3_shape = h3.get_shape().as_list()
            h3_reshape = tf.reshape(h3,
                                    [tf.shape(image)[0], h3_shape[1] * h3_shape[2] * self.df_dim * 8])
            # [fully, bn, lrelu]
            # h4 = lrelu(bn(linear(h3_reshape, self.dfc_dim, 'd_h3_lin'), training=training))
            # fully
            h4 = linear(h3_reshape, 1, 'd_h4_lin')
            return h4

    # ===================================================
    # -----------------Network design--------------------
    # ===================================================
    @overrides(Model)
    def build_model(self):
        # Input place holder
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name="keep_probability")
        self.images = tf.placeholder(
            tf.float32, [None, self.image_height, self.image_width, self.image_dim], name='real_images')
        self.conditions = tf.placeholder(
            tf.float32, [None, self.condition_height, self.condition_width, self.condition_dim], name='real_conditions')
        self.labels = tf.placeholder(
            tf.int32, [None, self.image_height, self.image_width], name='real_conditions')
        # Generator and discriminator
        with tf.variable_scope("GEN"):
            self.fake_images, self.fake_labels = self.generator(self.conditions, training=self.is_training)
        with tf.variable_scope("DIS"):
            self.D_logits_real = self.discriminator(self.images, self.conditions,
                                                    reuse=False, training=self.is_training)
            self.D_logits_fake = self.discriminator(self.fake_images, self.conditions,
                                                    reuse=True, training=self.is_training)
        # Adversarial loss
        self.loss_G_adv, self.loss_D_adv = self.loss_wgan_gp()
        # Generate loss
        self.loss_generate = tf.reduce_mean(tf.abs(self.images - self.fake_images))
        # Classify loss
        self.loss_classify = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.fake_labels, labels=self.labels, name="entropy")))

        self.g_loss = self.loss_G_adv + self.lambda_loss_generate * self.loss_generate + \
            self.lambda_loss_classify * self.loss_classify
        self.d_loss = self.loss_D_adv
        # Variables and saver
        self.g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='GEN')
        self.d_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='DIS')
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.saver = tf.train.Saver(max_to_keep=2)

    def loss_wgan_gp(self):
        batch_size = tf.shape(self.images)[0]
        image_all_dim = self.image_height * self.image_width * self.image_dim
        real_data = tf.reshape(self.images, [batch_size, image_all_dim])
        fake_data = tf.reshape(self.fake_images, [batch_size, image_all_dim])
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
        return gen_cost, disc_cost

    # ===================================================
    # -----------------Training phase--------------------
    # ===================================================
    @overrides(Model)
    def train(self):
        # Training optimizer
        with tf.control_dependencies(self.extra_update_ops):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2) \
                .minimize(self.d_loss, var_list=self.d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2) \
                .minimize(self.g_loss, var_list=self.g_vars)
        # Training summary logs
        if self.debug:
            self.summary_op = tf.summary.merge([
                tf.summary.histogram("histogram/D_real", self.D_logits_real),
                tf.summary.histogram("histogram/D_fake", self.D_logits_fake),
                tf.summary.histogram("histogram/G", self.fake_images),

                tf.summary.scalar("loss/loss_generate", self.loss_generate),
                tf.summary.scalar("loss/loss_classify", self.loss_classify),
                tf.summary.scalar("loss/d_loss", self.d_loss),
                tf.summary.scalar("loss/g_loss", self.g_loss),
            ])
            self.writer = tf.summary.FileWriter("./{}_logs".format(self.name), self.sess.graph)
        # Initialize and restore model
        tf.global_variables_initializer().run(session=self.sess)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            counter = 0
            print(" [!] Load failed...")
        self.train_update(counter)

    @overrides(Model)
    def train_update(self, counter=0):
        # Get samples
        images_sample, conditions_sample, labels_sample = self.get_sample()
        sample_feed = {self.is_training: False, self.keep_prob: 1.0,
                       self.images: images_sample, self.conditions: conditions_sample, self.labels: labels_sample}
        start_time = time.time()
        # every epoch
        for epoch in range(self.epoch):
            np.random.shuffle(self.data)
            batch_idxs = min(len(self.data), self.train_size) // self.batch_size
            # every mini-batch
            for idx in range(0, batch_idxs):
                # Get feeds
                batch_files = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_images, batch_conditions, batch_labels = self.get_batch(batch_files)
                batch_feed = {self.is_training: True, self.keep_prob: 0.8,
                              self.images: batch_images, self.conditions: batch_conditions, self.labels: batch_labels}
                # Update network
                _ = self.sess.run(self.d_optim, feed_dict=batch_feed)
                _ = self.sess.run(self.g_optim, feed_dict=batch_feed)
                _ = self.sess.run(self.g_optim, feed_dict=batch_feed)
                # Training logs
                err_d, err_g_adv, err_g_generate, err_g_class = self.sess.run(
                    [self.d_loss, self.loss_G_adv, self.loss_generate, self.loss_classify], feed_dict=batch_feed)
                print("Epoch: [%2d] [%4d/%4d] [%7d] time: %4.4f, "
                      "d_loss: %.8f, err_g_adv: %.8f, err_g_generate: %.8f, err_g_class: %.8f" %
                      (epoch, idx, batch_idxs, counter, time.time() - start_time,
                       err_d, err_g_adv, err_g_generate, err_g_class))
                if np.mod(counter, 5) == 0:
                    summary_str = self.sess.run(self.summary_op, feed_dict=batch_feed)
                    self.writer.add_summary(summary_str, counter)
                # Sample logs and visualized images
                if np.mod(counter, 100) == 0:
                    g_samples, g_labels_samples, d_loss_samples, g_loss_samples, g_loss_generate_samples, g_loss_classify_samples = \
                        self.sess.run([self.fake_images, self.fake_labels, self.d_loss,
                                       self.loss_G_adv, self.loss_generate, self.loss_classify], feed_dict=sample_feed)
                    images_visual = self.output_visual(g_samples)
                    scipy.misc.imsave('./{}/train_gen_{:06d}.png'.format(self.sample_dir, counter),
                                      images_visual.astype(np.uint8))
                    labels_visual = self.label_visual(g_labels_samples)
                    scipy.misc.imsave('./{}/train_classify_{:06d}.png'.format(self.sample_dir, counter),
                                      labels_visual.astype(np.uint8))
                    print("[Sample] d_loss: %.8f, err_g_adv: %.8f, err_g_generate: %.8f, err_g_class: %.8f"
                          % (d_loss_samples, g_loss_samples, g_loss_generate_samples, g_loss_classify_samples))
                # Checkpoint, save model
                if np.mod(counter, 500) == 400:
                    print('Model saved...')
                    self.save(self.checkpoint_dir, counter)

                counter += 1

    # ===================================================
    # -----------------Testing phase---------------------
    # ===================================================
    def generate(self):
        tf.global_variables_initializer().run(session=self.sess)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
            result_dir = './results'
            visual_dir = './visual'
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            if not os.path.exists(visual_dir):
                os.makedirs(visual_dir)

            # TODO: generate a batch?
            for idx in range(0, len(self.data_val)):
                # Get feeds
                batch_files = self.data_val[idx]
                image_files, conditions_files = zip(batch_files)
                batch_images = get_batch_images_norm(image_files, self.need_flip, scale=1.0)
                batch_conditions = get_batch_images_norm(conditions_files, self.need_flip, scale=1.0)
                name = image_files[0].split('/')[-1]
                # Feed
                sample_feed = {self.is_training: False, self.keep_prob: 1.0,
                               self.images: batch_images, self.conditions: batch_conditions}
                samples_g, err_d, err_g, err_g1, err_g2 = self.sess.run(
                    [self.fake_images, self.d_loss, self.g_loss, self.g1_loss, self.g2_loss], feed_dict=sample_feed)
                print("[%.20s %4d/%4d] d_loss: %.8f, g_loss: %.8f, g1_loss: %.8f, g2_loss: %.8f" %
                      (name, idx, len(self.data_val), err_d, err_g, err_g1, err_g2))
                # Save
                scipy.misc.imsave('./{}/{}'.format(result_dir, name),
                                  merge(denorm_image(samples_g), [1, 1]).astype(np.uint8))
                scipy.misc.imsave('./{}/{}'.format(visual_dir, name),
                                  merge(denorm_image(batch_conditions), [1, 1]).astype(np.uint8))
                # break
        else:
            print(" [!] Load failed...")
            raise Exception("[!] Train a model first, then run test mode")
