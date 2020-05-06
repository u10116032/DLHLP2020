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
  dataset = dataset_builder.build(filenames, prefetch=2, batch=4)

  auto_encoder = AutoEncoder()
  discriminator = Discriminator(num_speaker=2)
  optimizer = tf.keras.optimizers.Adam()
  huber_loss = tf.keras.losses.Huber()
  crossentropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
      label_smoothing=0.2)

  train_loss = tf.keras.metrics.Mean(name='train_loss')
  ckpt = tf.train.Checkpoint( auto_encoder=auto_encoder,
                              discriminator=discriminator,
                              optimizer=optimizer )

  def train_step(origin, speaker_one_hot, target_wav, epoch_rate):
    with tf.GradientTape() as tape:
      encoded, decoded = auto_encoder(origin, speaker_one_hot)
      logits = discriminator(encoded)
      rec_loss = huber_loss(decoded, target_wav)
      d_loss = crossentropy_loss(speaker_one_hot, logits)
      loss = rec_loss + (-0.01) * epoch_rate * d_loss
      trainable_variables = auto_encoder.trainable_variables + \
          discriminator.trainable_variables
      gradients = tape.gradient(loss, trainable_variables)
      optimizer.apply_gradients(zip(gradients, trainable_variables))
    train_loss(loss)

  latest_ckpt = tf.train.latest_checkpoint(args.logdir)
  if latest_ckpt is not None:
    dummy_voice = np.ones((1, 512, audio_process.n_mels, 1))
    dummy_speaker = np.eye(2)[[0]]
    train_step(dummy_voice, dummy_speaker, dummy_voice, 0)
    ckpt.restore(latest_ckpt).assert_consumed()
  ckpt_mgr = tf.train.CheckpointManager(ckpt, args.logdir, max_to_keep=5)

  EPOCHS = 10000
  for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()

    for feature in tqdm(dataset):
      speaker_id = feature['speaker_id']
      speaker_one_hot = tf.one_hot(speaker_id, 2)
      mel = feature['mel']
      mel = tf.expand_dims(mel, axis=-1)
      try:
        train_step(mel, speaker_one_hot, mel, epoch/EPOCHS)
      except Exception as e:
        print(e)
        print(mel.shape)
    ckpt_mgr.save()
    template = 'Epoch {}, Loss: {}'
    print(template.format(epoch + 1, train_loss.result()))
  ckpt_mgr.save()

if __name__=='__main__':
  main()
