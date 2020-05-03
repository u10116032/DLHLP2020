import numpy as np
import tensorflow as tf
import argparse
import glob
import os
import re
import audio_process
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Path to dataset folder',
    required=True)

def audio_example(filepath):
  filename = filepath.strip().split('/')[-1]
  speaker_id, sample_id = re.match(r'p(\d+)_(\d+)\.wav', filename).groups()
  mag, mel = audio_process.wav2spectrogram(filepath)
  mag = np.transpose(mag.astype(np.float32))
  mel = np.transpose(mel.astype(np.float32))

  feature = {
      'speaker_id': tf.train.Feature(int64_list=tf.train.Int64List(
          value=[int(speaker_id)])),
      'sample_id': tf.train.Feature(int64_list=tf.train.Int64List(
          value=[int(sample_id)])),
      'filename': tf.train.Feature(bytes_list=tf.train.BytesList(
          value=[str.encode(filename)])),
      'mel': tf.train.Feature(bytes_list=tf.train.BytesList(
          value=[mel.tostring()])),
      'mel_shape': tf.train.Feature(int64_list=tf.train.Int64List(
          value=list(mel.shape))),
      'mag': tf.train.Feature(bytes_list=tf.train.BytesList(
          value=[mag.tostring()])),
      'mag_shape': tf.train.Feature(int64_list=tf.train.Int64List(
          value=list(mag.shape)))
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

def main():
  args = parser.parse_args()
  dataset = args.dataset
  files = glob.glob(os.path.join(dataset, '*/*.wav'))

  groups = defaultdict(lambda: [])
  for file in files:
    filename = file.strip().split('/')[-1]
    speaker_id, sample_id = re.match(r'p(\d+)_(\d+)\.wav', filename).groups()
    groups[speaker_id].append(file)

  for speaker, files in groups.items():
    tfrecord_filename = "p%s.tfrecord" % speaker
    with tf.io.TFRecordWriter(tfrecord_filename) as writer:
        for file in files:
          tf_example = audio_example(file)
          writer.write(tf_example.SerializeToString())
          print(tf_example)

  print(groups)


if __name__ == '__main__':
  main()
