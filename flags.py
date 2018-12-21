import tensorflow as tf

flags = tf.app.flags


flags.DEFINE_integer("time_start", 0, "The time we start to process ")
flags.DEFINE_integer("time_steps", 540, "The number of values after time_start we want to take as feature")
flags.DEFINE_list("struct", [300, 100, 32], "The network structure")
flags.DEFINE_bool("use_RNN", False, "Whether to use RNN")
flags.DEFINE_integer("batch_size", 64, "Batch_size")
flags.DEFINE_integer("epochs", 50, "Epochs")
flags.DEFINE_float("learning_rate", 0.1, "Learning_rate")
