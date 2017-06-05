from __future__ import division
from model import Model
from glob import glob
import time
from utils import *
from ops import *


class GenerateCity(Model):
    def __init__(self, sess, config):
        # init
        super(GenerateCity, self).__init__(sess=sess, config=config)
        # Network architect
        self.num_of_class = 34

    # ===================================================
    # -----------------Files processing------------------
    # ===================================================
    @overrides(Model)
    def get_training_data(self):
        data = glob(os.path.join(self.image_dir, self.input_fname_pattern))
        return data

    @overrides(Model)
    def get_valid_data(self):
        data = glob(os.path.join(self.image_dir_val, self.input_fname_pattern))
        return data

    @overrides(Model)
    def get_sample(self):
        # Get sample
        images, conditions = get_city_generate(self.sample_files, self.condition_dir_val, self.need_flip)
        sample_images, sample_conditions = images, conditions
        # Sample visual
        images_visual = merge(denorm_image(sample_images), [self.manifold_h_sample, self.manifold_w_sample])
        scipy.misc.imsave('./{}/sample_0_images.png'.format(self.sample_dir), images_visual.astype(np.uint8))
        conditions_visual = merge(label_id_visual(sample_conditions), [self.manifold_h_sample, self.manifold_w_sample])
        scipy.misc.imsave('./{}/sample_2_conditions.png'.format(self.sample_dir), conditions_visual.astype(np.uint8))
        # Images and conditions blending
        images_visual = np.array(images_visual * 0.5 + conditions_visual * 0.5).astype(np.float32)
        images_visual[np.nonzero(images_visual > 255.)] = 255.
        scipy.misc.imsave('./{}/sample_1_image_condition.png'.format(self.sample_dir),
                          images_visual.astype(np.uint8))

        return sample_images, sample_conditions

    @overrides(Model)
    def get_batch(self, files):
        images, conditions = get_city_generate(files, self.condition_dir)
        return images, conditions

    @overrides(Model)
    def output_visual(self, data):
        images_visual = merge(denorm_image(data), [self.manifold_h_sample, self.manifold_w_sample])
        return images_visual

    # ===================================================
    # -----------------Generator-------------------------
    # ===================================================
    def generator(self, condition, training=True, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            condition = tf.cast(tf.expand_dims(condition, axis=3), dtype=tf.float32)

            # condition is (256 x 512 x input_c_dim)
            e1 = conv2d(condition, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 256 x self.gf_dim)
            e2 = bn(conv2d(lrelu(e1), self.gf_dim * 2, name='g_e2_conv'), training=training)
            # e2 is (64 x 128 x self.gf_dim*2)
            e3 = bn(conv2d(lrelu(e2), self.gf_dim * 4, name='g_e3_conv'), training=training)
            # e3 is (32 x 64 x self.gf_dim*4)
            e4 = bn(conv2d(lrelu(e3), self.gf_dim * 8, name='g_e4_conv'), training=training)
            # e4 is (16 x 32 x self.gf_dim*8)
            e5 = bn(conv2d(lrelu(e4), self.gf_dim * 8, name='g_e5_conv'), training=training)
            # e5 is (8 x 16 x self.gf_dim*8)
            e6 = bn(conv2d(lrelu(e5), self.gf_dim * 8, name='g_e6_conv'), training=training)
            # e6 is (4 x 8 x self.gf_dim*8)
            e7 = bn(conv2d(lrelu(e6), self.gf_dim * 8, name='g_e7_conv'), training=training)
            # e7 is (2 x 4 x self.gf_dim*8)
            e8 = conv2d(lrelu(e7), self.gf_dim * 8, name='g_e8_conv')
            # e8 is (1 x 2 x self.gf_dim*8)

            deconv_shape1 = e7.get_shape().as_list()
            d1 = bn(deconv2d(tf.nn.relu(e8), [-1, deconv_shape1[1], deconv_shape1[2], self.gf_dim * 8],
                             name='g_d1'), training=training)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)
            deconv_shape2 = e6.get_shape().as_list()
            d2 = bn(deconv2d(tf.nn.relu(d1), [-1, deconv_shape2[1], deconv_shape2[2], self.gf_dim * 8],
                             name='g_d2'), training=training)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)
            deconv_shape3 = e5.get_shape().as_list()
            d3 = bn(deconv2d(tf.nn.relu(d2), [-1, deconv_shape3[1], deconv_shape3[2], self.gf_dim * 8],
                             name='g_d3'), training=training)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)
            deconv_shape4 = e4.get_shape().as_list()
            d4 = bn(deconv2d(tf.nn.relu(d3), [-1, deconv_shape4[1], deconv_shape4[2], self.gf_dim * 8],
                             name='g_d4'), training=training)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)
            deconv_shape5 = e3.get_shape().as_list()
            d5 = bn(deconv2d(tf.nn.relu(d4), [-1, deconv_shape5[1], deconv_shape5[2], self.gf_dim * 4],
                             name='g_d5'), training=training)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)
            deconv_shape6 = e2.get_shape().as_list()
            d6 = bn(deconv2d(tf.nn.relu(d5), [-1, deconv_shape6[1], deconv_shape6[2], self.gf_dim * 2],
                             name='g_d6'), training=training)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)
            deconv_shape7 = e1.get_shape().as_list()
            d7 = bn(deconv2d(tf.nn.relu(d6), [-1, deconv_shape7[1], deconv_shape7[2], self.gf_dim],
                             name='g_d7'), training=training)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)
            deconv_shape8 = condition.get_shape().as_list()
            d8 = deconv2d(tf.nn.relu(d7), [-1, deconv_shape8[1], deconv_shape8[2], self.image_dim], name='g_d8')
            # d8 is (256 x 256 x output_c_dim)
            return tf.nn.tanh(d8)

    def discriminator(self, image, condition, reuse=False, training=True):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            condition = tf.cast(tf.expand_dims(condition, axis=3), dtype=tf.float32)

            image = tf.concat([image, condition], 3)
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # [conv, bn, lrelu] x 3
            condition_s1 = max_pool_2x2(condition)
            h0 = tf.concat([h0, condition_s1], 3)
            h1 = lrelu(bn(conv2d(h0, self.df_dim * 2, name='d_h1_conv'), training=training))

            condition_s2 = max_pool_2x2(condition_s1)
            h1 = tf.concat([h1, condition_s2], 3)
            h2 = lrelu(bn(conv2d(h1, self.df_dim * 4, name='d_h2_conv'), training=training))

            condition_s3 = max_pool_2x2(condition_s2)
            h2 = tf.concat([h2, condition_s3], 3)
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
            tf.int32, [None, self.condition_height, self.condition_width], name='real_conditions')
        # Generator and discriminator
        with tf.variable_scope("GEN"):
            self.fake_images = self.generator(self.conditions, training=self.is_training)
        with tf.variable_scope("DIS"):
            self.D_logits_real = self.discriminator(self.images, self.conditions,
                                                    reuse=False, training=self.is_training)
            self.D_logits_fake = self.discriminator(self.fake_images, self.conditions, reuse=True, training=self.is_training)
        self.g_loss, self.d_loss = self.loss()
        # Variables and saver
        self.g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='GEN')
        self.d_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='DIS')
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.saver = tf.train.Saver(max_to_keep=2)

    def loss(self):
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

        loss_g1 = tf.reduce_mean(tf.abs(self.images - self.fake_images))
        gen_cost += self.lambda_g1 * loss_g1
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
                _ = self.sess.run(self.d_optim, feed_dict=batch_feed)
                _ = self.sess.run(self.g_optim, feed_dict=batch_feed)
                _ = self.sess.run(self.g_optim, feed_dict=batch_feed)
                # Training logs
                err_d, err_g = self.sess.run([self.d_loss, self.g_loss], feed_dict=batch_feed)
                print("Epoch: [%2d] [%4d/%4d] [%7d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" %
                      (epoch, idx, batch_idxs, counter, time.time() - start_time, err_d, err_g))
                if np.mod(counter, 5) == 0:
                    summary_str = self.sess.run(self.summary_op, feed_dict=batch_feed)
                    self.writer.add_summary(summary_str, counter)
                # Sample logs and visualized images
                if np.mod(counter, 100) == 0:
                    g_samples, d_loss_samples, g_loss_samples = \
                        self.sess.run([self.fake_images, self.d_loss, self.g_loss], feed_dict=sample_feed)
                    images_visual = self.output_visual(g_samples)
                    scipy.misc.imsave('./{}/train_{:06d}.png'.format(self.sample_dir, counter),
                                      images_visual.astype(np.uint8))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss_samples, g_loss_samples))
                # Checkpoint, save model
                if np.mod(counter, 500) == 100:
                    print('Model saved...')
                    self.save(self.checkpoint_dir, counter)

                counter += 1

    # ===================================================
    # -----------------Testing phase---------------------
    # ===================================================
    @overrides(Model)
    def pred(self):
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

            val_conditions = []
            for folder in os.listdir(self.condition_dir_val):
                path = os.path.join(self.condition_dir_val, folder, "*.png")
                val_conditions.extend(glob(path))
            # TODO: test data batch?
            for idx in range(0, len(val_conditions)):
                # Get feeds
                batch_files = val_conditions[idx]
                name = batch_files.split('/')[-1]
                print('{:d}/{:d}: {}'.format(idx, len(val_conditions), name))
                batch_image = [scipy.misc.imread(batch_files).astype(np.uint8)]
                sample_feed = {self.is_training: False, self.keep_prob: 1.0, self.conditions: batch_image}
                # Feed
                samples_g = self.sess.run(self.fake_images, feed_dict=sample_feed)
                samples_g_out = np.argmax(np.squeeze(samples_g, axis=0), axis=2)
                # Save
                scipy.misc.imsave('./{}/{}'.format(result_dir, name), samples_g_out.astype(np.uint8))
                label_v = label_id_visual_single(samples_g_out)
                scipy.misc.imsave('./{}/{}'.format(visual_dir, name), label_v.astype(np.uint8))
                # break
        else:
            print(" [!] Load failed...")
            raise Exception("[!] Train a model first, then run test mode")
