from model import AutoEncoder, Discriminator
import tensorflow as tf
import numpy as np
import dataset_builder
import argparse
import yaml
import glob
import os
from tqdm import tqdm
from utils import audio_process

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Path to dataset folder',
    required=True)
parser.add_argument('--logdir', help='Log Path', default='./ckpt')


def main():
  args = parser.parse_args()
  filenames = glob.glob(os.path.join(args.dataset,'*.tfrecord'))
  dataset = dataset_builder.build(filenames, prefetch=2, batch=2)

  auto_encoder = AutoEncoder()
  auto_encoder_optimizer = tf.keras.optimizers.Adam(0.0001)
  discriminator = Discriminator(num_speaker=2)
  discriminator_optimizer = tf.keras.optimizers.Adam(0.0001)
  huber_loss = tf.keras.losses.Huber()
  crossentropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
      label_smoothing=0.2)

  auto_encoder_metric = tf.keras.metrics.Mean(name='auto_encoder_loss')
  discriminator_metric = tf.keras.metrics.Mean(name='discriminator_loss')
  ckpt = tf.train.Checkpoint( auto_encoder=auto_encoder,
                              discriminator=discriminator,
                              auto_encoder_optimizer=auto_encoder_optimizer,
                              discriminator_optimizer=discriminator_optimizer )

  def train_D(origin, speaker_id, target_wav):
    with tf.GradientTape() as tape:
      encoded, decoded = auto_encoder(origin, speaker_id-1)
      logits = discriminator(encoded)
      speaker_one_hot = tf.one_hot(speaker_id, 2)
      d_loss = crossentropy_loss(speaker_one_hot, logits)
      discriminator_gradients = tape.gradient(d_loss,
        discriminator.trainable_variables)
      discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables))
    discriminator_metric(d_loss)

  def train_AE(origin, speaker_id, target_wav, epoch_rate):
    with tf.GradientTape() as tape:
      encoded, decoded = auto_encoder(origin, speaker_id-1)
      logits = discriminator(encoded)
      rec_loss = huber_loss(decoded, target_wav)
      speaker_one_hot = tf.one_hot(speaker_id, 2)
      d_loss = crossentropy_loss(speaker_one_hot, logits)
      auto_encoder_loss = rec_loss + (-0.01) * epoch_rate * d_loss
      auto_encoder_gradients = tape.gradient(auto_encoder_loss,
        auto_encoder.trainable_variables)
      auto_encoder_optimizer.apply_gradients(
        zip(auto_encoder_gradients, auto_encoder.trainable_variables))
    auto_encoder_metric(auto_encoder_loss)

  latest_ckpt = tf.train.latest_checkpoint(args.logdir)
  if latest_ckpt is not None:
    dummy_voice = np.ones((1, 512, audio_process.n_mels, 1), dtype=np.float32)
    dummy_speaker = np.eye(2,dtype=np.float32)[[0]]
    train_D(dummy_voice, dummy_speaker, dummy_voice)
    train_AE(dummy_voice, dummy_speaker, dummy_voice, 0)
    ckpt.restore(latest_ckpt).assert_consumed()
  ckpt_mgr = tf.train.CheckpointManager(ckpt, args.logdir, max_to_keep=5)

  EPOCHS = 100
  for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    auto_encoder_metric.reset_states()
    discriminator_metric.reset_states()

    for idx, feature in enumerate(tqdm(dataset)):
      speaker_id = feature['speaker_id']
      mel = feature['mel']
      mel = tf.expand_dims(mel, axis=-1)
      if idx%5 == 0:
        train_AE(mel, speaker_id, mel, epoch/EPOCHS)
      else:
        train_D(mel, speaker_id, mel)
    ckpt_mgr.save()
    template = 'Epoch {}, AE Loss: {}, D Loss: {}'
    log_msg = template.format( epoch+1,
                               auto_encoder_metric.result(),
                               discriminator_metric.result() )
    print(log_msg)
  ckpt_mgr.save()

if __name__=='__main__':
  main()
