import numpy as np
from model_ac import ACCity
from utils_lite import pp, show_all_variables, config_check_directory
import tensorflow as tf

flags = tf.app.flags
# Experiments setting
flags.DEFINE_string("name", "cityscapes_(classify)_(ac)", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("mode", "train", "network mode [train, test]")
flags.DEFINE_integer("sample_num", 16, "The number of sample images [64]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_boolean("debug", True, "True for debugging, False for nothing [False]")
# Network feed
flags.DEFINE_string("dataset_name", "CITYSCAPES_DATASET", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("image_dir", "./dataset/CITYSCAPES_DATASET/train/semantic_color", "The directory of images")
flags.DEFINE_string("condition_dir", "./dataset/CITYSCAPES_DATASET/train/image", "The directory of conditions")
flags.DEFINE_string("label_dir", "./dataset/CITYSCAPES_DATASET/train/semantic_id", "The directory of conditions")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
flags.DEFINE_integer("image_height", 128, "The size of the images to produce [64]")
flags.DEFINE_integer("image_width", 256, "The size of the images to produce.")
flags.DEFINE_integer("image_dim", 3, "The dimension of the image [3]")
flags.DEFINE_integer("condition_height", 128, "The size of the condition [256]")
flags.DEFINE_integer("condition_width", 256, "The size of the condition.")
flags.DEFINE_integer("condition_dim", 3, "The dimension of the condition [3]")
flags.DEFINE_integer("num_of_class", 34, "The size of the number of class [34]")
# Network sample(testing) feed
flags.DEFINE_string("image_dir_val", "./dataset/CITYSCAPES_DATASET/val/semantic_color", "The directory of images")
flags.DEFINE_string("condition_dir_val", "./dataset/CITYSCAPES_DATASET/val/image", "The directory of conditions")
flags.DEFINE_string("label_dir_val", "./dataset/CITYSCAPES_DATASET/val/semantic_id", "The directory of conditions")
# Hyper parameter
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("beta2", 0.999, "Momentum term of adam [0.5]")
flags.DEFINE_float("lambda_gp", 10.0, "Gradient penalty lambda hyper parameter [10.0]")
flags.DEFINE_float("lambda_loss_generate", 20.0, "L1 lambda hyper parameter [10.0]")
flags.DEFINE_float("lambda_loss_classify", 80.0, "L1 lambda hyper parameter [10.0]")
# Directories
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("test_dir", "test", "Directory name to save the image samples [test]")
FLAGS = flags.FLAGS


def main(_):
    # Config directories check
    config_check_directory(FLAGS)
    pp.pprint(flags.FLAGS.__flags)
    # Run config
    run_config = tf.ConfigProto()
    run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    # Build model
    with tf.Session(config=run_config) as sess:
        model = ACCity(sess, FLAGS)
        model.build_model()
    show_all_variables()
    # launch mode
    if FLAGS.mode == 'train':
        model.train()
    elif FLAGS.mode == 'generate':
        model.generate()

if __name__ == '__main__':
    tf.app.run()