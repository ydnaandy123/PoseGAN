"""
fully convolutional dense net with wgan_gp
"""
import numpy as np
from model_fcn import FCN
from utils import pp, show_all_variables, config_check
import tensorflow as tf

flags = tf.app.flags
# Main
flags.DEFINE_string("name", "cityscapes_(classify)_(generate)all_label_(condition)image_(fcn)", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("mode", "pred", "network mode [train, test]")
flags.DEFINE_integer("sample_num", 1, "The number of sample images [64]")
flags.DEFINE_integer("batch_size", 9, "The size of batch images [64]")
flags.DEFINE_boolean("default_setting", True, "True for default setting, False for nothing [False]")
# Auto
flags.DEFINE_string("dataset_name", "CITYSCAPES_DATASET", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("image_dir", "./dataset/CITYSCAPES_DATASET/train/semantic_id", "The directory of images")
flags.DEFINE_string("condition_dir", "./dataset/CITYSCAPES_DATASET/train/image", "The directory of conditions")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
flags.DEFINE_integer("image_height", 64, "The size of the ima7ges to produce [64]")
flags.DEFINE_integer("image_width", 64, "The size of the images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("image_dim", 3, "The dimension of the image [3]")
flags.DEFINE_integer("condition_height", 256, "The size of the condition [256]")
flags.DEFINE_integer("condition_width", 256, "The size of the condition. If None, same value as output_height [None]")
flags.DEFINE_integer("condition_dim", 3, "The dimension of the condition [3]")

flags.DEFINE_boolean("need_condition", True, "True for conditional, False for nothing [False]")
flags.DEFINE_boolean("need_g1", False, "True for plus g1, False for nothing [False]")
flags.DEFINE_boolean("classify", False, "True for classify, False for generate [False]")
flags.DEFINE_string("g1_mode", "pix2pix", "The directory of conditions")
# Details
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("beta2", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_float("lambda_gp", 10.0, "Gradient penalty lambda hyper parameter [10.0]")
flags.DEFINE_float("lambda_g1", 100.0, "L1 lambda hyper parameter [10.0]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("debug", True, "True for debugging, False for nothing [False]")
# Directories
flags.DEFINE_string("vgg_dir", "./checkpoint", "Directory name to vgg_net [samples]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("test_dir", "test", "Directory name to save the image samples [test]")
FLAGS = flags.FLAGS


def main(_):
    # Flags
    config, model_dir = config_check(FLAGS, default_setting=FLAGS.default_setting)
    pp.pprint(flags.FLAGS.__flags)
    # Run config
    run_config = tf.ConfigProto()
    run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    # Build model
    with tf.Session(config=run_config) as sess:
        model = FCN(sess, config)
    show_all_variables()
    # launch mode
    if FLAGS.mode == 'train':
        model.train(config)
    elif FLAGS.mode == 'test':
        model.test()
    elif FLAGS.mode == 'pred':
        model.pred()

if __name__ == '__main__':
    tf.app.run()
