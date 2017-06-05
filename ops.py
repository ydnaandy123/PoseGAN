import tensorflow as tf


def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x, name=name)


def elu(x, name='elu'):
    return tf.nn.elu(x, name=name)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def conv2d_basic(x, w, bias):
    conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv


def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
    batch_size = tf.shape(input_)[0]
    deconv_shape = tf.stack([batch_size] + output_shape[1:])

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=deconv_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    input_dim = input_.get_shape().as_list()[1]
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_dim, output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def bn(inputs, training=True, momentum=0.9, epsilon=1e-5):
    return tf.layers.batch_normalization(inputs, training=training, momentum=momentum, epsilon=epsilon)


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            scale=True, is_training=train, scope=self.name)

###
# FCN
###
def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))

"""
" fully convolutional dense net
"""


def bn_relu_conv(inputs, n_filters, filter_size=3, keep_prob=0.8, training=True, name='bn_relu_conv'):
    """
    Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0) on the inputs
    """
    # TODO: dropout order?
    l = conv2d(elu(bn(inputs, training=training)),
               output_dim=n_filters, k_h=filter_size, k_w=filter_size, d_h=1, d_w=1, name=name)

    #l = conv2d(elu(inputs),
    #           output_dim=n_filters, k_h=filter_size, k_w=filter_size, d_h=1, d_w=1, name=name)

    l = tf.nn.dropout(l, keep_prob=keep_prob)
    return l


def transition_down(inputs, n_filters, keep_prob=0.8, training=True, name='transition_down'):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """

    l = bn_relu_conv(inputs, n_filters, filter_size=1, keep_prob=keep_prob, training=training, name=name)
    # l = Pool2DLayer(l, 2, mode='max')
    # TODO: remove pooling?
    l = avg_pool_2x2(l)

    return l
    # Note : network accuracy is quite similar with average pooling or without BN - ReLU.
    # We can also reduce the number of parameters reducing n_filters in the 1x1 convolution


def transition_up(skip_connection, block_to_upsample, n_filters_keep, training=True, name='transition_yp'):
    """
    Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection """

    # Upsample
    l = tf.concat(block_to_upsample, 3)
    output_shape = l.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[2] *= 2
    output_shape[3] = n_filters_keep

    l = bn(deconv2d(elu(l), output_shape, name=name, k_h=3, k_w=3), training=training)
    #l = deconv2d(elu(l), output_shape, name=name, k_h=3, k_w=3)
    # Concatenate with skip connection
    l = tf.concat([l, skip_connection], 3)

    return l
    # Note : we also tried Subpixel Deconvolution without seeing any improvements.
    # We can reduce the number of parameters reducing n_filters_keep in the Deconvolution

