from model import Generator, Discriminator, DomainClassifier
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

def main():
  args = parser.parse_args()
  filenames = glob.glob(os.path.join(args.dataset,'*.tfrecord'))
  speaker_count = len(filenames)
  x_dataset = dataset_builder.build(filenames, prefetch=128, batch=4)
  y_dataset = dataset_builder.build(filenames, prefetch=128, batch=4)
  D, G, C = Discriminator(), Generator(), DomainClassifier()
  G_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)
  D_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)
  C_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5, beta_2=0.999)

  ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
      label_smoothing=0.1)
  # huber_loss = tf.keras.losses.Huber()
  l1_loss = tf.keras.losses.MeanAbsoluteError()

  G_metric = tf.keras.metrics.Mean(name='G_loss')
  D_metric = tf.keras.metrics.Mean(name='D_loss')
  C_metric = tf.keras.metrics.Mean(name='C_loss')

  summary_writer = tf.summary.create_file_writer(args.logdir)

  def train_G(real_x, real_x_attr, real_y, real_y_attr):
    with tf.GradientTape() as tape:
      fake_y = G(real_x, real_y_attr)
      reconst_x = G(fake_y, real_x_attr)
      fake_y_d = D(fake_y, real_y_attr)
      fake_y_c = C(fake_y)
      fake_x = G(real_x, real_x_attr)
      gan_loss = tf.reduce_mean(-1 * ops.safe_log(fake_y_d))
      cycle_loss = l1_loss(real_x, reconst_x)
      cls_loss = ce_loss(real_y_attr, fake_y_c)
      identity_loss = l1_loss(real_x, fake_x)
      loss = gan_loss + 2 * cycle_loss + cls_loss + 2 * identity_loss
      G_gradients = tape.gradient(loss, G.trainable_variables)
      G_optimizer.apply_gradients(zip(G_gradients, G.trainable_variables))
    tf.summary.scalar('loss_G_gan_loss', gan_loss, step=G_optimizer.iterations)
    tf.summary.scalar('loss_G_cycle_loss', cycle_loss, step=G_optimizer.iterations)
    tf.summary.scalar('loss_G_cls_loss', cls_loss, step=G_optimizer.iterations)
    tf.summary.scalar('loss_G_identity_loss', identity_loss, step=G_optimizer.iterations)
    G_metric(loss)
    return loss

  def train_D(real_x, real_x_attr, real_y, real_y_attr):
    with tf.GradientTape() as tape:
      fake_y = G(real_x, real_y_attr)
      fake_y_d = D(fake_y, real_y_attr)
      real_y_d = D(real_y, real_y_attr)
      gan_loss = tf.reduce_mean(
          -1 * ops.safe_log(real_y_d) + (-1) * ops.safe_log(1 - fake_y_d))
      loss = gan_loss
      D_gradients = tape.gradient(loss, D.trainable_variables)
      D_optimizer.apply_gradients(zip(D_gradients, D.trainable_variables))
    D_metric(loss)
    return loss

  def train_C(real_x, real_x_attr, real_y, real_y_attr):
    with tf.GradientTape() as tape:
      real_y_c = C(real_y)
      loss = ce_loss(real_y_attr, real_y_c)
      C_gradients = tape.gradient(loss, C.trainable_variables)
      C_optimizer.apply_gradients(zip(C_gradients, C.trainable_variables))
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

  EPOCHS = 3000
  with summary_writer.as_default():
    for epoch in range(EPOCHS):
      G_metric.reset_states()
      D_metric.reset_states()
      C_metric.reset_states()

      for idx, features in tqdm(enumerate(zip(x_dataset, y_dataset))):
        x_feature, y_feature = features
        x_id = np.asarray(x_feature['speaker_id'], dtype=np.int32)
        x_id_onehot = tf.one_hot(x_id, speaker_count)
        x = tf.expand_dims(x_feature['mfcc'], axis=-1)

        y_id = np.asarray(y_feature['speaker_id'], dtype=np.int32)
        y_id_onehot = tf.one_hot(y_id, speaker_count)
        y = tf.expand_dims(y_feature['mfcc'], axis=-1)
        loss_D = train_D(x, x_id_onehot, y, y_id_onehot)
        loss_C = train_C(x, x_id_onehot, y, y_id_onehot)
        loss_G = train_G(x, x_id_onehot, y, y_id_onehot)
        tf.summary.scalar('loss_D', loss_D, step=D_optimizer.iterations)
        tf.summary.scalar('loss_C', loss_C, step=C_optimizer.iterations)
        tf.summary.scalar('loss_G', loss_G, step=G_optimizer.iterations)
      ckpt_mgr.save()
      template = 'Epoch {}, G Loss: {}, D Loss: {}, C Loss: {}'
      log_msg = template.format( epoch+1,
                                 G_metric.result(),
                                 D_metric.result(),
                                 C_metric.result() )
      print(log_msg)
    ckpt_mgr.save()


if __name__ == '__main__':
  main()
