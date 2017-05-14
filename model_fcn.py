import tensorflow as tf
import scipy.sparse
import scipy.misc
import numpy as np
import os
import fcn_ops


class FCN(object):
    def __init__(self, flags, num_of_class):
        self.FLAGS = flags
        self.NUM_OF_CLASSESS = num_of_class

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
                kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
                if self.FLAGS.debug:
                    utils.add_activation_summary(current)
            elif kind == 'pool':
                current = utils.avg_pool_2x2(current)
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
        # model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL) # If not sure whether the model existed or not
        model_data = scipy.io.loadmat(os.path.join(self.FLAGS.vgg_dir, 'imagenet-vgg-verydeep-19.mat'))

        mean = model_data['normalization'][0][0][0]
        mean_pixel = np.mean(mean, axis=(0, 1))
        processed_image = fcn_ops.process_image(image, mean_pixel)

        weights = np.squeeze(model_data['layers'])

        with tf.variable_scope("inference"):
            image_net = self.vgg_net(weights, processed_image)
            conv_final_layer = image_net["conv5_3"]

            pool5 = fcn_ops.max_pool_2x2(conv_final_layer)

            w6 = fcn_ops.weight_variable([7, 7, 512, 4096], name="W6")
            b6 = fcn_ops.bias_variable([4096], name="b6")
            conv6 = fcn_ops.conv2d_basic(pool5, w6, b6)
            relu6 = tf.nn.relu(conv6, name="relu6")
            if self.FLAGS.debug:
                fcn_ops.add_activation_summary(relu6)
            relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

            w7 = fcn_ops.weight_variable([1, 1, 4096, 4096], name="W7")
            b7 = fcn_ops.bias_variable([4096], name="b7")
            conv7 = fcn_ops.conv2d_basic(relu_dropout6, w7, b7)
            relu7 = tf.nn.relu(conv7, name="relu7")
            if self.FLAGS.debug:
                fcn_ops.add_activation_summary(relu7)
            relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

            w8 = fcn_ops.weight_variable([1, 1, 4096, self.NUM_OF_CLASSESS], name="W8")
            b8 = fcn_ops.bias_variable([self.NUM_OF_CLASSESS], name="b8")
            conv8 = fcn_ops.conv2d_basic(relu_dropout7, w8, b8)
            # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

            # now to upscale to actual image size
            deconv_shape1 = image_net["pool4"].get_shape()
            w_t1 = fcn_ops.weight_variable([4, 4, deconv_shape1[3].value, self.NUM_OF_CLASSESS], name="W_t1")
            b_t1 = fcn_ops.bias_variable([deconv_shape1[3].value], name="b_t1")
            conv_t1 = fcn_ops.conv2d_transpose_strided(conv8, w_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
            fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

            deconv_shape2 = image_net["pool3"].get_shape()
            w_t2 = fcn_ops.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
            b_t2 = fcn_ops.bias_variable([deconv_shape2[3].value], name="b_t2")
            conv_t2 = fcn_ops.conv2d_transpose_strided(fuse_1, w_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
            fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

            shape = tf.shape(image)
            deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], self.NUM_OF_CLASSESS])
            w_t3 = fcn_ops.weight_variable([16, 16, self.NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
            b_t3 = fcn_ops.bias_variable([self.NUM_OF_CLASSESS], name="b_t3")
            conv_t3 = fcn_ops.conv2d_transpose_strided(fuse_2, w_t3, b_t3, output_shape=deconv_shape3, stride=8)

            annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

        return tf.expand_dims(annotation_pred, dim=3), conv_t3