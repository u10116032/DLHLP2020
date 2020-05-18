import tensorflow as tf

def safe_log(x, eps=1e-12):
  return tf.math.log(x + eps)
