"""
Adopt DCGAN to train lsp
"""
import numpy as np
from model_conditional import DCGAN_conditional
from utils import pp, show_all_variables, config_check
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("dataset_name", "condition_pose_compare", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("mode", "train", "network mode [train, test]")

flags.DEFINE_string("image_dir", "/data/vllab1/pose-hg-train/data/LSP/train/annot", "The directory of images")
flags.DEFINE_string("condition_dir", "/data/vllab1/pose-hg-train/data/LSP/train/images", "The directory of conditions")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")

flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("test_dir", "test", "Directory name to save the image samples [samples]")

flags.DEFINE_integer("sample_num", 16, "The number of sample images [64]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")

flags.DEFINE_integer("image_height", 64, "The size of the ima7ges to produce [64]")
flags.DEFINE_integer("image_width", 64, "The size of the images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("image_dim", 16, "The dimension of the image [3]")
flags.DEFINE_integer("condition_height", 256, "The size of the condition [256]")
flags.DEFINE_integer("condition_width", 256, "The size of the condition. If None, same value as output_height [None]")
flags.DEFINE_integer("condition_dim", 3, "The dimension of the condition [3]")

flags.DEFINE_integer("epoch", 20, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    config, model_dir = config_check(FLAGS)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    run_config.gpu_options.per_process_gpu_memory_fraction = 1.0

    with tf.Session(config=run_config) as sess:
        dcgan_conditional = DCGAN_conditional(sess, config, needCondition=False)
    show_all_variables()

    if FLAGS.mode == 'train':
        dcgan_conditional.train(config)
    elif FLAGS.mode == 'test':
        dcgan_conditional.test(config)
    else:
        if not dcgan_conditional.load(FLAGS.checkpoint_dir)[0]:
            raise Exception("[!] Train a model first, then run test mode")

if __name__ == '__main__':
    tf.app.run()
