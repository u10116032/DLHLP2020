from model import Generator, Discriminator, DomainClassifier
from utils.mcep_normalizer import McepNormalizer
from utils import audio_process
from utils import ops
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
import sys
import glob
import dataset_builder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Path to dataset folder',
    required=True)
parser.add_argument('--logdir', help='Log Path', default='./ckpt')

learning_rate = 0.002
total_steps = 150000
warmup_learning_rate = 0.0
warmup_steps = 1000

def main():
  args = parser.parse_args()
  filenames = glob.glob(os.path.join(args.dataset,'*.tfrecord'))
  speaker_count = len(filenames)
  x_dataset = dataset_builder.build(filenames, prefetch=128, batch=8)
  y_dataset = dataset_builder.build(filenames, prefetch=128, batch=8)
  D, G, C = Discriminator(), Generator(), DomainClassifier()
  G_lr = tf.Variable(0.0, dtype=tf.float64)
  G_optimizer = tf.keras.optimizers.Adam(G_lr, beta_1=0.5, beta_2=0.999)
  D_lr = tf.Variable(0.0, dtype=tf.float64)
  D_optimizer = tf.keras.optimizers.Adam(D_lr, beta_1=0.5, beta_2=0.999)
  C_lr = tf.Variable(0.0, dtype=tf.float64)
  C_optimizer = tf.keras.optimizers.Adam(C_lr, beta_1=0.5, beta_2=0.999)
  # G_optimizer = tf.keras.optimizers.SGD(0.0001)
  # D_optimizer = tf.keras.optimizers.SGD(0.0001)
  # C_optimizer = tf.keras.optimizers.SGD(0.0001)

  ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, 
      label_smoothing=0.1)
  # huber_loss = tf.keras.losses.Huber()
  l1_loss = tf.keras.losses.MeanAbsoluteError()

  G_metric = tf.keras.metrics.Mean(name='G_loss')
  D_metric = tf.keras.metrics.Mean(name='D_loss')
  C_metric = tf.keras.metrics.Mean(name='C_loss')

  summary_writer = tf.summary.create_file_writer(args.logdir)

  def train_G(real_x, real_x_attr, real_y, real_y_attr, step):
    with tf.GradientTape() as tape:
      fake_y = G(real_x, real_y_attr, True)
      reconst_x = G(fake_y, real_x_attr, True)
      fake_y_d = D(fake_y, real_y_attr, False)
      fake_y_c = C(fake_y, False)
      fake_x = G(real_x, real_x_attr, True)
      gan_loss = tf.reduce_mean(-1 * ops.safe_log(fake_y_d))
      # gan_loss = tf.reduce_mean(-1 * fake_y_d)
      cycle_loss = l1_loss(real_x, reconst_x)
      cls_loss = ce_loss(real_y_attr, fake_y_c)
      identity_loss = l1_loss(real_x, fake_x)
      loss = gan_loss + 10 * cycle_loss + cls_loss + 5 * identity_loss
      G_gradients = tape.gradient(loss, G.trainable_variables)
      G_optimizer.apply_gradients(zip(G_gradients, G.trainable_variables))
    G_lr.assign(ops.cosine_lr( step, learning_rate, total_steps, 
                               warmup_learning_rate, warmup_steps))
    tf.summary.scalar('loss_G/gan_loss', gan_loss, step=G_optimizer.iterations)
    tf.summary.scalar('loss_G/cycle_loss', cycle_loss, step=G_optimizer.iterations)
    tf.summary.scalar('loss_G/cls_loss', cls_loss, step=G_optimizer.iterations)
    tf.summary.scalar('loss_G/identity_loss', identity_loss, step=G_optimizer.iterations)
    G_metric(loss)
    return loss

  def train_D(real_x, real_x_attr, real_y, real_y_attr, step):
    with tf.GradientTape() as tape:
      fake_y = G(real_x, real_y_attr, False)
      fake_y_d = D(fake_y, real_y_attr, True)
      real_y_d = D(real_y, real_y_attr, True)
      gan_loss = tf.reduce_mean(
          -1 * ops.safe_log(real_y_d) + (-1) * ops.safe_log(1 - fake_y_d))
      # gan_loss = 0.5 * tf.reduce_mean(fake_y_d - real_y_d)

      # alpha = tf.random.uniform([tf.shape(fake_y)[0],1,1,1], 0, 1)
      # inter = alpha * real_x + (1-alpha) * fake_y
      with tf.GradientTape() as t:
          t.watch(fake_y)
          pred = D(fake_y, real_y_attr, True)
      grad = t.gradient(pred, [fake_y])[0]
      slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1,2,3]))
      gradient_penalty = tf.reduce_mean((slopes - 1.)**2)

      loss = gan_loss # + 10 * gradient_penalty
      D_gradients = tape.gradient(loss, D.trainable_variables)
      D_optimizer.apply_gradients(zip(D_gradients, D.trainable_variables))
    D_lr.assign(ops.cosine_lr( step, learning_rate, total_steps, 
                               warmup_learning_rate, warmup_steps))
    tf.summary.scalar('loss_D/gan_loss', gan_loss, step=D_optimizer.iterations)
    tf.summary.scalar('loss_D/negative_critic_loss', -1*gan_loss, step=D_optimizer.iterations)
    tf.summary.scalar('loss_D/gp', gradient_penalty, step=D_optimizer.iterations)
    D_metric(loss)
    return loss

  def train_C(real_x, real_x_attr, real_y, real_y_attr, step):
    with tf.GradientTape() as tape:
      real_y_c = C(real_y, True)
      loss = ce_loss(real_y_attr, real_y_c)
      C_gradients = tape.gradient(loss, C.trainable_variables)
      C_optimizer.apply_gradients(zip(C_gradients, C.trainable_variables))
    C_lr.assign(ops.cosine_lr( step, learning_rate, total_steps, 
                               warmup_learning_rate, warmup_steps))
    C_metric(loss)
    return loss

  ckpt = tf.train.Checkpoint( D=D,
                              G=G,
                              C=C,
                              G_optimizer=G_optimizer,
                              D_optimizer=D_optimizer,
                              C_optimizer=C_optimizer )
  latest_ckpt = tf.train.latest_checkpoint(args.logdir)
  if latest_ckpt is not None:
    dummy_wav = np.zeros((1, 36, 512, 1), dtype=np.float32)
    dummy_speaker_onehot = np.zeros((1, speaker_count), dtype=np.float32)
    train_D(dummy_wav, dummy_speaker_onehot, dummy_wav, dummy_speaker_onehot)
    train_C(dummy_wav, dummy_speaker_onehot, dummy_wav, dummy_speaker_onehot)
    train_G(dummy_wav, dummy_speaker_onehot, dummy_wav, dummy_speaker_onehot)
    ckpt.restore(latest_ckpt).assert_consumed()
  ckpt_mgr = tf.train.CheckpointManager(ckpt, args.logdir, max_to_keep=5)

  with summary_writer.as_default():
    G_metric.reset_states()
    D_metric.reset_states()
    C_metric.reset_states()

    step = 0
    mcep_normalizer = McepNormalizer('./train/norm_dict.pkl')
    for features in tqdm(zip(x_dataset, y_dataset), total=total_steps):
      x_feature, y_feature = features

      x_id = np.asarray(x_feature['speaker_id'], dtype=np.int32)
      x_id_onehot = tf.one_hot(x_id - 1, speaker_count)
      x = mcep_normalizer.batch_mcep_norm(x_feature['mcep'], x_id)
      x = tf.expand_dims(x, axis=-1)

      y_id = np.asarray(y_feature['speaker_id'], dtype=np.int32)
      y_id_onehot = tf.one_hot(y_id - 1, speaker_count)
      y = mcep_normalizer.batch_mcep_norm(y_feature['mcep'], y_id)
      y = tf.expand_dims(y, axis=-1)

      if (step+1) % 2 == 0:
        loss_G = train_G(x, x_id_onehot, y, y_id_onehot, step)
        tf.summary.scalar('loss_G', loss_G, step=G_optimizer.iterations)
      else:
        loss_D = train_D(x, x_id_onehot, y, y_id_onehot, step)
        loss_C = train_C(x, x_id_onehot, y, y_id_onehot, step)
        tf.summary.scalar('loss_D', loss_D, step=D_optimizer.iterations)
        tf.summary.scalar('loss_C', loss_C, step=C_optimizer.iterations)
      if (step+1)%100 == 0:
        ckpt_mgr.save()
        template = 'Steps {}, G Loss: {}, D Loss: {}, C Loss: {}'
        log_msg = template.format( step,
                                   G_metric.result(),
                                   D_metric.result(),
                                   C_metric.result() )
        print(log_msg)
      if step >= total_steps:
        break
      step += 1
    print('Finish Trainin, saving ckpt...')
    template = 'Steps {}, G Loss: {}, D Loss: {}, C Loss: {}'
    log_msg = template.format( step,
                               G_metric.result(),
                               D_metric.result(),
                               C_metric.result() )
    ckpt_mgr.save()
    print('Done.')


if __name__ == '__main__':
  main()
