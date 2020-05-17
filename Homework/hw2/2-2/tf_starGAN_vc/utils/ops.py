import tensorflow as tf

def safe_log(x, sanity=1e-7):
  return tf.math.log(x+sanity)
