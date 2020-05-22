import tensorflow as tf
import numpy as np

def safe_log(x, eps=1e-12):
  return tf.math.log(x + eps)

def cosine_lr( global_step,
               learning_rate_base,
               total_steps,
               warmup_learning_rate=0.0,
               warmup_steps=0):
  global_step = tf.cast(global_step, dtype=tf.float64)
  learning_rate_base = tf.cast(learning_rate_base, dtype=tf.float64)
  total_steps = tf.cast(total_steps, dtype=tf.float64)
  warmup_learning_rate = tf.cast(warmup_learning_rate, dtype=tf.float64)
  warmup_steps = tf.cast(warmup_steps, dtype=tf.float64)

  decay_ratio = (global_step - warmup_steps) / (total_steps - warmup_steps)
  learning_rate = 0.5 * learning_rate_base * (1 + tf.math.cos(np.pi*decay_ratio))
  if warmup_steps > 0:
    slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
    warmup_rate = slope * global_step + warmup_learning_rate
    learning_rate = tf.where( global_step < warmup_steps,
                              warmup_rate,
                              learning_rate)
  return learning_rate
