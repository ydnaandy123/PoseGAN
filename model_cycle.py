from __future__ import division
from model import Model
from glob import glob
import time
from utils_lite import *
from ops import *


class CycleCity(Model):
    def __init__(self, sess, config, gf_dim=64, df_dim=64):
        # init
        super(CycleCity, self).__init__(sess=sess, config=config)
        # Hyper parameter
        self.epoch = config.epoch
        self.learning_rate = config.learning_rate
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.lambda_gp = config.lambda_gp
        self.lambda_rec = config.lambda_rec
        self.lambda_con = config.lambda_con
        # Network architect
        self.num_of_class = 34
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        # ===================================================
        # -----------------Model graph-----------------------
        # ===================================================
        # Feed
        self.is_training = self.keep_prob = None
        self.images = self.conditions = None
        self.batch_size_realtime = self.image_all_dim = self.image_shape = None
        # GEN, DIS
        self.fake_images_B = self.fake_conditions_BA = None
        self.fake_conditions_A = self.fake_images_AB = None
        self.D_logits_real_B = self.D_logits_fake_B = None
        self.D_logits_real_A = self.D_logits_fake_A = None
        # ===================================================
        # -----------------Loss------------------------------
        # ===================================================
        # Loss Adv
        self.loss_G_adv_B = self.loss_D_adv_B = self.loss_G_adv_A = self.loss_D_adv_A = None
        # Loss Rec
        self.loss_rec = self.loss_rec_a = self.loss_rec_b = None
        # Loss Con (supervise)
        self.loss_con_B = self.loss_con_A = None
        # Loss All
        self.loss_G_B = self.loss_G_A = self.loss_D_B = self.loss_D_A = None
        # Training phase
        self.extra_update_ops = self.saver = None
        self.g_vars_A = self.d_vars_A = self.g_vars_B = self.d_vars_B = None
        self.d_optim_A = self.g_optim_A = self.d_optim_B = self.g_optim_B = self.summary_op = self.writer = None

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
        data = zip(data, data_condition)
        return data

    @overrides(Model)
    def get_valid_data(self):
        data = sorted(glob(os.path.join(self.image_dir_val, self.input_fname_pattern)))
        data_condition = sorted(glob(os.path.join(self.condition_dir_val, self.input_fname_pattern)))
        data = zip(data, data_condition)
        return data

    @overrides(Model)
    def get_sample(self):
        # Get sample
        image_files, conditions_files = zip(*self.sample_files)
        sample_images = get_batch_images_norm(image_files, self.need_flip, scale=1.0)
        sample_conditions = get_batch_images_norm(conditions_files, self.need_flip, scale=1.0)
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

        return sample_images, sample_conditions

    @overrides(Model)
    def get_batch(self, files):
        # Get batch
        image_files, conditions_files = zip(*files)
        images = get_batch_images_norm(image_files, self.need_flip, scale=0.5)
        conditions = get_batch_images_norm(conditions_files, self.need_flip, scale=0.5)
        return images, conditions

    @overrides(Model)
    def output_visual(self, data):
        images_visual = merge(denorm_image(data), [self.manifold_h_sample, self.manifold_w_sample])
        return images_visual

    # ===================================================
    # -----------------Generator-------------------------
    # ===================================================
    def generator(self, condition, training=True, reuse=False, name='generator'):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            # condition is (128 x 256 x input_c_dim)
            e1 = conv2d(condition, self.gf_dim, name='g_e1_conv')
            # e1 is (64 x 128 x self.gf_dim)
            e2 = bn(conv2d(elu(e1), self.gf_dim * 2, name='g_e2_conv'), training=training)
            # e2 is (32 x 64 x self.gf_dim*2)
            e3 = bn(conv2d(elu(e2), self.gf_dim * 4, name='g_e3_conv'), training=training)
            # e3 is (16 x 32 x self.gf_dim*4)
            e4 = bn(conv2d(elu(e3), self.gf_dim * 8, name='g_e4_conv'), training=training)
            # e4 is (8 x 16 x self.gf_dim*8)
            e5 = bn(conv2d(elu(e4), self.gf_dim * 8, name='g_e5_conv'), training=training)
            # e5 is (4 x 8 x self.gf_dim*8)
            e6 = bn(conv2d(elu(e5), self.gf_dim * 8, name='g_e6_conv'), training=training)
            # e6 is (2 x 4 x self.gf_dim*8)
            e7 = bn(conv2d(elu(e6), self.gf_dim * 8, name='g_e7_conv'), training=training)
            # e7 is (1 x 2 x self.gf_dim*8)
            e8 = conv2d(elu(e7), self.gf_dim * 8, name='g_e8_conv')
            # e8 is (1 x 1 x self.gf_dim*8)

            d1 = bn(deconv2d(elu(e8), [-1, self.s_h128, self.s_w128, self.gf_dim * 8],
                             name='g_d1'), training=training)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (1 x 2 x self.gf_dim*8*2)
            d2 = bn(deconv2d(elu(d1), [-1, self.s_h64, self.s_w64, self.gf_dim * 8],
                             name='g_d2'), training=training)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (2 x 4 x self.gf_dim*8*2)
            d3 = bn(deconv2d(elu(d2), [-1, self.s_h32, self.s_w32, self.gf_dim * 8],
                             name='g_d3'), training=training)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (4 x 8 x self.gf_dim*8*2)
            d4 = bn(deconv2d(elu(d3), [-1, self.s_h16, self.s_w16, self.gf_dim * 8],
                             name='g_d4'), training=training)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (8 x 16 x self.gf_dim*8*2)
            d5 = bn(deconv2d(elu(d4), [-1, self.s_h8, self.s_w8, self.gf_dim * 4],
                             name='g_d5'), training=training)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (16 x 32 x self.gf_dim*4*2)
            d6 = bn(deconv2d(elu(d5), [-1, self.s_h4, self.s_w4, self.gf_dim * 2],
                             name='g_d6'), training=training)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (32 x 64 x self.gf_dim*2*2)
            d7 = bn(deconv2d(elu(d6), [-1, self.s_h2, self.s_w2, self.gf_dim],
                             name='g_d7'), training=training)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (64 x 128 x self.gf_dim*1*2)
            d8 = deconv2d(elu(d7), [-1, self.s_h, self.s_w, self.image_dim], name='g_d8')
            # d8 is (128 x 256 x output_c_dim)
            return tf.nn.tanh(d8)

    def discriminator(self, image, condition, reuse=False, training=True, name='discriminator'):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            image = tf.concat([image, condition], 3)
            h0 = elu(conv2d(image, self.df_dim, name='d_h0_conv'))

            h1 = elu(bn(conv2d(h0, self.df_dim * 2, name='d_h1_conv'), training=training))

            h2 = elu(bn(conv2d(h1, self.df_dim * 4, name='d_h2_conv'), training=training))

            h3 = elu(bn(conv2d(h2, self.df_dim * 8, name='d_h3_conv'), training=training))

            h3_shape = h3.get_shape().as_list()
            h3_reshape = tf.reshape(h3,
                                    [tf.shape(image)[0], h3_shape[1] * h3_shape[2] * self.df_dim * 8])

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
        # Size
        self.batch_size_realtime = tf.shape(self.images)[0]
        self.image_all_dim = self.image_height * self.image_width * self.image_dim
        self.image_shape = [self.batch_size_realtime, self.image_all_dim]
        # Generator and discriminator

        with tf.variable_scope("GEN"):
            self.fake_images_B = self.generator(self.conditions, reuse=False, training=self.is_training, name='A2B')
            self.fake_conditions_BA = self.generator(
                self.fake_images_B, reuse=False, training=self.is_training, name='B2A')

            self.fake_conditions_A = self.generator(self.images, reuse=True, training=self.is_training, name='B2A')
            self.fake_images_AB = self.generator(
                self.fake_conditions_A, reuse=True, training=self.is_training, name='A2B')
        with tf.variable_scope("DIS"):
            self.D_logits_real_B = self.discriminator(
                self.images, self.conditions, reuse=False, training=self.is_training, name='B')
            self.D_logits_fake_B = self.discriminator(
                self.fake_images_B, self.conditions, reuse=True, training=self.is_training, name='B')

            self.D_logits_real_A = self.discriminator(
                self.conditions, self.images, reuse=False, training=self.is_training, name='A')
            self.D_logits_fake_A = self.discriminator(
                self.fake_conditions_A, self.images, reuse=True, training=self.is_training, name='A')

        # Adversarial Loss
        self.loss_G_adv_B, self.loss_D_adv_B = self.loss_adversarial_b()
        self.loss_G_adv_A, self.loss_D_adv_A = self.loss_adversarial_a()
        # Cycle loss
        self.loss_rec = self.loss_reconstruction()
        # Conditional (supervise) loss
        self.loss_con_B = tf.reduce_mean(tf.abs(self.images - self.fake_images_B))
        self.loss_con_A = tf.reduce_mean(tf.abs(self.conditions - self.fake_conditions_A))
        # All loss
        self.loss_G_B = self.loss_G_adv_B + self.lambda_rec * self.loss_rec + self.lambda_con * self.loss_con_B
        self.loss_D_B = self.loss_D_adv_B
        self.loss_G_A = self.loss_G_adv_A + self.lambda_rec * self.loss_rec + self.lambda_con * self.loss_con_A
        self.loss_D_A = self.loss_D_adv_A

        # Variables and saver
        self.g_vars_B = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='GEN/A2B')
        self.d_vars_B = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='DIS/B')
        self.g_vars_A = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='GEN/B2A')
        self.d_vars_A = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='DIS/A')
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.saver = tf.train.Saver(max_to_keep=2)

    def loss_reconstruction(self):
        ab = tf.reduce_mean(tf.abs(self.images - self.fake_images_AB))
        ba = tf.reduce_mean(tf.abs(self.conditions - self.fake_conditions_BA))
        self.loss_rec_a = ab
        self.loss_rec_b = ba
        return ab + ba

    def loss_adversarial_b(self):
        real_data = tf.reshape(self.images, self.image_shape)
        disc_real = self.D_logits_real_B
        fake_data = tf.reshape(self.fake_images_B, self.image_shape)
        disc_fake = self.D_logits_fake_B
        # Standard WGAN loss
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        # Gradient penalty
        alpha = tf.random_uniform(shape=[self.batch_size_realtime, 1], minval=0., maxval=1.)
        differences = fake_data - real_data
        interpolates = real_data + (alpha * differences)
        interpolates = tf.reshape(interpolates, [self.batch_size_realtime, self.image_height, self.image_width, self.image_dim])
        with tf.variable_scope("DIS"):
            # TODO: why [interpolates]?
            d_logits_ = self.discriminator(
                interpolates, self.conditions, reuse=True, training=self.is_training, name='B')
            gradients = tf.gradients(d_logits_, [interpolates])[0]
        gradients = tf.reshape(gradients, self.image_shape)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        disc_cost += self.lambda_gp * gradient_penalty
        return gen_cost, disc_cost

    def loss_adversarial_a(self):
        real_data = tf.reshape(self.conditions, self.image_shape)
        disc_real = self.D_logits_real_A
        fake_data = tf.reshape(self.fake_conditions_A, self.image_shape)
        disc_fake = self.D_logits_fake_A
        # Standard WGAN loss
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        # Gradient penalty
        alpha = tf.random_uniform(shape=[self.batch_size_realtime, 1], minval=0., maxval=1.)
        differences = fake_data - real_data
        interpolates = real_data + (alpha * differences)
        interpolates = tf.reshape(interpolates, [self.batch_size_realtime, self.image_height, self.image_width, self.image_dim])
        with tf.variable_scope("DIS"):
            # TODO: why [interpolates]?
            d_logits_ = self.discriminator(
                interpolates, self.conditions, reuse=True, training=self.is_training, name='A')
            gradients = tf.gradients(d_logits_, [interpolates])[0]
        gradients = tf.reshape(gradients, self.image_shape)
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
            self.g_optim_B = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2) \
                .minimize(self.loss_G_B, var_list=self.g_vars_B)
            self.d_optim_B = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2) \
                .minimize(self.loss_D_B, var_list=self.d_vars_B)
            self.g_optim_A = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2) \
                .minimize(self.loss_G_A, var_list=self.g_vars_A)
            self.d_optim_A = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2) \
                .minimize(self.loss_D_A, var_list=self.d_vars_A)
        # Training summary logs
        if self.debug:
            self.summary_op = tf.summary.merge([
                tf.summary.histogram("histogram/AB", self.fake_images_AB),
                tf.summary.histogram("histogram/B", self.fake_images_B),
                tf.summary.histogram("histogram/A", self.fake_conditions_A),
                tf.summary.histogram("histogram/BA", self.fake_conditions_BA),

                tf.summary.scalar("loss/g_B_loss", self.loss_G_B),
                tf.summary.scalar("loss/d_B_loss", self.loss_D_B),
                tf.summary.scalar("loss/g_A_loss", self.loss_G_A),
                tf.summary.scalar("loss/d_A_loss", self.loss_D_A),
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
        images_sample, conditions_sample = self.get_sample()
        sample_feed = {self.is_training: False, self.keep_prob: 1.0,
                       self.images: images_sample, self.conditions: conditions_sample}
        start_time = time.time()
        # every epoch
        for epoch in range(self.epoch):
            np.random.shuffle(self.data)
            batch_idxs = min(len(self.data), self.train_size) // self.batch_size
            # every mini-batch
            for idx in range(0, batch_idxs):
                # Get feeds
                batch_files = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_images, batch_conditions = self.get_batch(batch_files)
                batch_feed = {self.is_training: True, self.keep_prob: 0.8,
                              self.images: batch_images, self.conditions: batch_conditions}
                # Update network
                _ = self.sess.run(self.g_optim_B, feed_dict=batch_feed)
                _ = self.sess.run(self.d_optim_B, feed_dict=batch_feed)
                _ = self.sess.run(self.g_optim_A, feed_dict=batch_feed)
                _ = self.sess.run(self.d_optim_A, feed_dict=batch_feed)
                # Training logs
                err_g_adv_b, err_g_con_b, err_g_rec_b, err_g_adv_a, err_g_con_a, err_g_rec_a, err_d_a, err_d_b = \
                    self.sess.run([
                        self.loss_G_adv_B, self.loss_con_B, self.loss_rec_b,
                        self.loss_G_adv_A, self.loss_con_A, self.loss_rec_a,
                        self.loss_D_adv_B, self.loss_D_adv_A],
                        feed_dict=batch_feed)
                print("Epoch: [%2d] [%4d/%4d] [%7d] time: %4.4f, "
                      "err_g_adv_b: %.8f, err_g_con_b: %.8f, err_g_rec_b: %.8f,"
                      "err_g_adv_a: %.8f, err_g_con_a: %.8f, err_g_rec_a: %.8f,"
                      "err_d_a: %.8f, err_d_b: %.8f" %
                      (epoch, idx, batch_idxs, counter, time.time() - start_time,
                       err_g_adv_b, err_g_con_b, err_g_rec_b, err_g_adv_a, err_g_con_a, err_g_rec_a, err_d_a, err_d_b))
                if np.mod(counter, 5) == 0:
                    summary_str = self.sess.run(self.summary_op, feed_dict=batch_feed)
                    self.writer.add_summary(summary_str, counter)
                # Sample logs and visualized images
                if np.mod(counter, 100) == 0:
                    g_samples_image, g_samples_condition, err_loss_g_b, err_loss_d_b, err_loss_loss_g_a, err_loss_d_a = \
                        self.sess.run([self.fake_images_B, self.fake_conditions_A,
                                       self.loss_G_B, self.loss_D_B, self.loss_G_A, self.loss_D_A], feed_dict=sample_feed)
                    images_visual = self.output_visual(g_samples_image)
                    scipy.misc.imsave('./{}/train_image_{:06d}.png'.format(self.sample_dir, counter),
                                      images_visual.astype(np.uint8))
                    images_visual = self.output_visual(g_samples_condition)
                    scipy.misc.imsave('./{}/train_condition_{:06d}.png'.format(self.sample_dir, counter),
                                      images_visual.astype(np.uint8))
                    print("[Sample] err_loss_g_b: %.8f, err_loss_d_b: %.8f, "
                          "err_loss_loss_g_a: %.8f, err_loss_d_a: %.8f, " %
                          (err_loss_g_b, err_loss_d_b, err_loss_loss_g_a, err_loss_d_a))
                # Checkpoint, save model
                if np.mod(counter, 500) == 100:
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
                samples_g, err_d, err_g, err_g1= self.sess.run(
                    [self.fake_images_B, self.fake_conditions_A, self.d_loss_all, self.g_loss_all, self.loss_l1_all], feed_dict=sample_feed)
                print("[%.20s %4d/%4d] d_loss: %.8f, g_loss: %.8f, g1_loss: %.8f" %
                      (name, idx, len(self.data_val), err_d, err_g, err_g1))
                # Save
                scipy.misc.imsave('./{}/{}'.format(result_dir, name),
                                  merge(denorm_image(samples_g), [1, 1]).astype(np.uint8))
                scipy.misc.imsave('./{}/{}'.format(visual_dir, name),
                                  merge(denorm_image(batch_conditions), [1, 1]).astype(np.uint8))
                # break
        else:
            print(" [!] Load failed...")
            raise Exception("[!] Train a model first, then run test mode")
