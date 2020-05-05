from model import AutoEncoder, Discriminator
import tensorflow as tf
import dataset_builder
import argparse
import yaml
import glob
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Path to dataset folder',
    required=True)
parser.add_argument('--logdir', help='Log Path', default='./ckpt')


def main():
  args = parser.parse_args()
  filenames = glob.glob(os.path.join(args.dataset,'*.tfrecord'))
  dataset = dataset_builder.build(filenames, prefetch=1, batch=2)
  auto_encoder = AutoEncoder()
  discriminator = Discriminator(num_speaker=2)
  optimizer = tf.keras.optimizers.Adam()
  huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
  crossentropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  ckpt = tf.train.Checkpoint(model=auto_encoder)
  ckpt_mgr = tf.train.CheckpointManager(ckpt, args.logdir, max_to_keep=5)

  def train_step(origin, speaker_one_hot, target):
    with tf.GradientTape() as tape:
      encoded, decoded = auto_encoder(origin, speaker_one_hot)
      logits = discriminator(encoded)
      rec_loss = tf.reduce_mean(huber_loss(decoded, target_wav))
      speaker_id_one_hot = tf.one_hot(speaker_id, 2)
      d_loss = crossentropy_loss(speaker_id_one_hot, logits)
      loss = rec_loss + (-1)*logits
      trainable_variables = auto_encoder.trainable_variables + \
          discriminator.trainable_variables
      gradients = tape.gradient(loss, trainable_variables)
      optimizer.apply_gradients(zip(gradients, trainable_variables))
    train_loss(loss)

  EPOCHS = 1000
  for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()

    for feature in tqdm(dataset):
      speaker_id = feature['speaker_id']
      speaker_one_hot = tf.one_hot(speaker_id, 2)
      mel = feature['mel']
      mel = tf.expand_dims(mel, axis=-1)
      try:
        train_step(mel, speaker_one_hot, mel)
      except Exception as e:
        print(e)
        print(mel.shape)
    ckpt_mgr.save()
    template = 'Epoch {}, Loss: {}'
    print(template.format(epoch + 1, train_loss.result()))
  ckpt_mgr.save()

if __name__=='__main__':
  main()
