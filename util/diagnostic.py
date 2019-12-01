import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print("gpus =", gpus)