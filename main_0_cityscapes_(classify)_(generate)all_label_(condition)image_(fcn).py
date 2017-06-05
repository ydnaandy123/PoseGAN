import numpy as np
from model_fcn import FcnCity
from utils import pp, show_all_variables, config_check
import tensorflow as tf

flags = tf.app.flags
# Experiments setting
flags.DEFINE_string("name", "cityscapes_(classify)_(fcn_base)", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("mode", "train", "network mode [train, test]")
flags.DEFINE_integer("sample_num", 1, "The number of sample images [64]")
flags.DEFINE_integer("batch_size", 9, "The size of batch images [64]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_boolean("debug", True, "True for debugging, False for nothing [False]")
# Network feed
flags.DEFINE_string("dataset_name", "CITYSCAPES_DATASET", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("image_dir", "./dataset/CITYSCAPES_DATASET/train/semantic_id", "The directory of images")
flags.DEFINE_string("condition_dir", "./dataset/CITYSCAPES_DATASET/train/image", "The directory of conditions")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
flags.DEFINE_integer("image_height", 256, "The size of the images to produce [64]")
flags.DEFINE_integer("image_width", 512, "The size of the images to produce.")
flags.DEFINE_integer("image_dim", 1, "The dimension of the image [3]")
flags.DEFINE_integer("condition_height", 256, "The size of the condition [256]")
flags.DEFINE_integer("condition_width", 256, "The size of the condition.")
flags.DEFINE_integer("condition_dim", 3, "The dimension of the condition [3]")
# Hyper parameter
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_float("beta2", 0.999, "Momentum term of adam [0.5]")
flags.DEFINE_float("lambda_gp", 10.0, "Gradient penalty lambda hyper parameter [10.0]")
flags.DEFINE_float("lambda_g1", 100.0, "L1 lambda hyper parameter [10.0]")
# Directories
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("test_dir", "test", "Directory name to save the image samples [test]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)
    # Run config
    run_config = tf.ConfigProto()
    run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    # Build model
    with tf.Session(config=run_config) as sess:
        model = FcnCity(sess, FLAGS)
        model.build_model()
    show_all_variables()
    # launch mode
    if FLAGS.mode == 'train':
        model.train()
    elif FLAGS.mode == 'test':
        model.test()
    elif FLAGS.mode == 'pred':
        model.pred()

if __name__ == '__main__':
    tf.app.run()