import numpy as np
import tensorflow as tf
import argparse
import glob
import os
import re
import audio_process
from collections import defaultdict
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Path to dataset folder',
    required=True)

def audio_example(filepath):
  filename = filepath.strip().split('/')[-1]
  speaker_id, sample_id = re.match(r'p(\d+)_(\d+)\.wav', filename).groups()
  mag, mel = audio_process.wav2spectrogram(filepath)
  mfcc = audio_process.wav2mfcc(filepath)
  f0, _, _, mcep = audio_process.wav2mcep(filepath)

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
          value=list(mag.shape))),
      'mfcc': tf.train.Feature(bytes_list=tf.train.BytesList(
          value=[mfcc.tostring()])),
      'mfcc_shape': tf.train.Feature(int64_list=tf.train.Int64List(
          value=list(mfcc.shape))),
      'mcep': tf.train.Feature(bytes_list=tf.train.BytesList(
          value=[mcep.tostring()])),
      'mcep_shape': tf.train.Feature(int64_list=tf.train.Int64List(
          value=list(mcep.shape)))
  }

  return tf.train.Example(features=tf.train.Features(feature=feature)), f0, mcep

def main():
  args = parser.parse_args()
  dataset = args.dataset
  files = glob.glob(os.path.join(dataset, '*/*.wav'))

  groups = defaultdict(lambda: [])
  for file in files:
    filename = file.strip().split('/')[-1]
    speaker_id, sample_id = re.match(r'p(\d+)_(\d+)\.wav', filename).groups()
    groups[speaker_id].append(file)

  norm_dict = {}
  for speaker, files in groups.items():
    tfrecord_filename = "p%s.tfrecord" % speaker
    norm_dict[speaker] = {}
    with tf.io.TFRecordWriter(tfrecord_filename) as writer:
        log_f0_list = []
        mcep_list = []
        for file in files:
          tf_example, f0, mcep = audio_example(file)
          writer.write(tf_example.SerializeToString())
          log_f0_list.append(np.log(f0+1e-21))
          mcep_list.append(mcep)

        concated_log_f0 = np.concatenate(log_f0_list)
        concated_mcep = np.concatenate(mcep_list, axis=1)
        norm_dict[speaker]['log_f0_mean'] = concated_log_f0.mean()
        norm_dict[speaker]['log_f0_std'] = concated_log_f0.std()
        norm_dict[speaker]['mcep_mean'] = concated_mcep.mean(axis=1)
        norm_dict[speaker]['mcep_std'] = concated_mcep.std(axis=1)

  with open('norm_dict.pkl', 'wb') as file:
    pickle.dump(norm_dict, file)
  print(norm_dict)


if __name__ == '__main__':
  main()
