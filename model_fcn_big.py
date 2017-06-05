from __future__ import division
from model import Model
from glob import glob
import time
from utils import *
from ops import *


class FcnCityBig(Model):
    def __init__(self, sess, config):
        # init
        super(FcnCityBig, self).__init__(sess=sess, config=config)
        # Network architect
        self.num_of_class = 34
        self.vgg_dir = './checkpoint'

    # ===================================================
    # -----------------Files processing------------------
    # ===================================================
    @overrides(Model)
    def get_training_data(self):
        data = []
        for folder in os.listdir(self.image_dir):
            path = os.path.join(self.image_dir, folder, "*_labelIds.png")
            data.extend(glob(path))
        return data

    @overrides(Model)
    def get_valid_data(self):
        data = []
        for folder in os.listdir(self.image_dir_val):
            path = os.path.join(self.image_dir_val, folder, "*_labelIds.png")
            data.extend(glob(path))
        return data

    @overrides(Model)
    def get_sample(self):
        # Get sample
        images, conditions = get_city_classify_valid(self.sample_files, self.condition_dir_val, self.need_flip)
        sample_images, sample_conditions = images, conditions
        # Sample visual
        images_visual = merge(label_id_visual(sample_images), [self.manifold_h_sample, self.manifold_w_sample])
        scipy.misc.imsave('./{}/sample_0_images.png'.format(self.sample_dir), images_visual.astype(np.uint8))
        conditions_visual = merge(sample_conditions, [self.manifold_h_sample, self.manifold_w_sample])
        scipy.misc.imsave('./{}/sample_2_conditions.png'.format(self.sample_dir), conditions_visual.astype(np.uint8))
        # Images and conditions blending
        images_visual = np.array(images_visual * 0.5 + conditions_visual * 0.5).astype(np.float32)
        images_visual[np.nonzero(images_visual > 255.)] = 255.
        scipy.misc.imsave('./{}/sample_1_image_condition.png'.format(self.sample_dir),
                          images_visual.astype(np.uint8))

        return sample_images, sample_conditions

    @overrides(Model)
    def get_batch(self, files):
        images, conditions = get_city_classify_valid(files, self.condition_dir)
        return images, conditions

    @overrides(Model)
    def output_visual(self, data):
        images_visual = merge(label_visual(data), [self.manifold_h_sample, self.manifold_w_sample])
        return images_visual

    # ===================================================
    # -----------------Generator-------------------------
    # ===================================================
    def generator(self, condition, training=True, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            output_layer = self.inference(condition)
            return output_layer

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

    def inference(self, image):
        print("setting up vgg initialized conv layers ...")
        model_data = scipy.io.loadmat(os.path.join(self.vgg_dir, 'imagenet-vgg-verydeep-19.mat'))
        mean = model_data['normalization'][0][0][0]
        mean_pixel = np.mean(mean, axis=(0, 1))
        weights = np.squeeze(model_data['layers'])
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
            relu_dropout6 = tf.nn.dropout(relu6, keep_prob=self.keep_prob)

            w7 = tf.get_variable('w7', [1, 1, 4096, 4096], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b7 = tf.get_variable('b7', [4096], initializer=tf.constant_initializer(0.0))
            conv7 = conv2d_basic(relu_dropout6, w7, b7)
            relu7 = tf.nn.relu(conv7, name="relu7")
            if self.debug:
                add_activation_summary(relu7)
            relu_dropout7 = tf.nn.dropout(relu7, keep_prob=self.keep_prob)

            w8 = tf.get_variable('w8', [1, 1, 4096, self.num_of_class],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            b8 = tf.get_variable('b8', [self.num_of_class], initializer=tf.constant_initializer(0.0))
            conv8 = conv2d_basic(relu_dropout7, w8, b8)
            # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

            # now to upscale to actual image size
            # TODO: no relu? fuse? concat?
            deconv_shape1 = image_net["pool4"].get_shape().as_list()
            conv_t1 = deconv2d(conv8, [-1, deconv_shape1[1], deconv_shape1[2], deconv_shape1[3]],
                               k_h=4, k_w=4, name='conv_t1')
            fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

            deconv_shape2 = image_net["pool3"].get_shape()
            conv_t2 = deconv2d(fuse_1, [-1, deconv_shape2[1], deconv_shape2[2], deconv_shape2[3]],
                               k_h=4, k_w=4, name='conv_t2')
            fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

            shape = image.get_shape()
            conv_t3 = deconv2d(fuse_2, [-1, shape[1], shape[2], self.num_of_class],
                               k_h=16, k_w=16, d_h=8, d_w=8, name='conv_t3')

        return conv_t3

    # ===================================================
    # -----------------Network design--------------------
    # ===================================================
    @overrides(Model)
    def build_model(self):
        # Input place holder
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name="keep_probability")
        self.images = tf.placeholder(
            tf.int32, [None, self.image_height, self.image_width], name='real_images')
        self.conditions = tf.placeholder(
            tf.float32, [None, self.condition_height, self.condition_width, self.condition_dim], name='real_conditions')
        # Generator and discriminator
        with tf.variable_scope("GEN"):
            self.fake_images = self.generator(self.conditions, training=self.is_training)
        self.g_loss, self.d_loss = self.loss()
        # Variables and saver
        self.g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='GEN')
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.saver = tf.train.Saver(max_to_keep=2)

    def loss(self):
        loss_classify = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.fake_images, labels=self.images, name="entropy")))
        loss_d = tf.constant(0.)
        return loss_classify, loss_d

    # ===================================================
    # -----------------Training phase--------------------
    # ===================================================
    @overrides(Model)
    def train(self):
        # Training optimizer
        with tf.control_dependencies(self.extra_update_ops):
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2) \
                .minimize(self.g_loss, var_list=self.g_vars)
        # Training summary logs
        if self.debug:
            self.summary_op = tf.summary.merge([
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
                #break
        else:
            print(" [!] Load failed...")
            raise Exception("[!] Train a model first, then run test mode")
