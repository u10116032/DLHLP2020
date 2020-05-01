from model import AutoEncoder
import tensorflow as tf
import dataset_builder
import argparse
import yaml
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Path to dataset folder',
    required=True)
parser.add_argument('--log', help='Log Path')


def main():
  args = parser.parse_args()
  filenames = glob.glob(os.path.join(args.dataset,'*.tfrecord'))
  dataset = dataset_builder.build(filenames)
  model = AutoEncoder()
  optimizer = tf.keras.optimizers.Adam()
  loss_object = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  ckpt = tf.train.Checkpoint(model=model)
  ckpt_mgr = tf.train.CheckpointManager(ckpt, './ckpt', max_to_keep=3)

  def train_step(origin, speaker_one_hot, target):
    with tf.GradientTape() as tape:
      convert = model(origin, speaker_one_hot)
      loss = tf.reduce_mean(loss_object(convert, target))
      gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

  EPOCHS = 1
  for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()

    for feature in dataset:
      speaker_id = feature['speaker_id']
      speaker_one_hot = tf.one_hot(speaker_id, 2)
      mel = feature['mel']
      mel = tf.expand_dims(mel, axis=-1)
      train_step(mel, speaker_one_hot, mel)
      break
    ckpt_mgr.save()
    template = 'Epoch {}, Loss: {}'
    print(template.format(epoch + 1, train_loss.result()))
  ckpt_mgr.save()
  signatures = model.call.get_concrete_function(
      tf.TensorSpec(shape=[None, None, 80, 1], dtype=tf.float32, name="x"),
      tf.TensorSpec(shape=[None, 2], dtype=tf.float32, name="speaker_one_hot"))
  tf.keras.models.save_model(model, 'ckpt', save_format='tf')

if __name__=='__main__':
  main()
