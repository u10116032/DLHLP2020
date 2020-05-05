import tensorflow as tf

def build(filepaths, batch=16, prefetch=4):
  feature_desciption = {
    'speaker_id': tf.io.FixedLenFeature([], tf.int64),
    'sample_id': tf.io.FixedLenFeature([], tf.int64),
    'filename': tf.io.FixedLenFeature([], tf.string),
    'mel': tf.io.FixedLenFeature([], tf.string),
    'mel_shape': tf.io.FixedLenFeature([2,], tf.int64),
    'mag': tf.io.FixedLenFeature([], tf.string),
    'mag_shape': tf.io.FixedLenFeature([2,], tf.int64)
  }
  def _parse_feature(example_proto):
    feature = tf.io.parse_single_example(example_proto, feature_desciption)
    speaker_id = feature['speaker_id']
    sample_id = feature['sample_id']
    filename = feature['filename']
    mag_shape = feature['mag_shape']
    mag = tf.io.decode_raw(feature['mag'], tf.float32)
    mag = tf.reshape(mag, mag_shape)
    pad_size = 4 - mag_shape[0]%4
    mag = tf.pad(mag, [[0,pad_size],[0,0]])
    mel_shape = feature['mel_shape']
    mel = tf.io.decode_raw(feature['mel'], tf.float32)
    mel = tf.reshape(mel, mel_shape)
    mel = tf.pad(mel, [[0,pad_size],[0,0]])

    tensor_dict = {}
    tensor_dict['speaker_id'] = speaker_id
    tensor_dict['sample_id'] = sample_id
    tensor_dict['filename'] = filename
    tensor_dict['mag'] = mag
    tensor_dict['mel'] = mel
    return tensor_dict

  pad_shape = {}
  pad_shape['speaker_id'] = []
  pad_shape['sample_id'] = []
  pad_shape['filename'] = []
  pad_shape['mag'] = [None, None]
  pad_shape['mel'] = [None, None]

  dataset = tf.data.Dataset.from_tensor_slices(filepaths)
  dataset = dataset.interleave(lambda x:
      tf.data.TFRecordDataset(x).map(_parse_feature,
      num_parallel_calls=len(filepaths)),
      cycle_length=4, block_length=4)
  # dataset = dataset.repeat(-1)
  dataset = dataset.shuffle(buffer_size=batch * prefetch)
  dataset = dataset.padded_batch(batch_size=batch, padded_shapes=pad_shape,
      drop_remainder=True)
  dataset = dataset.prefetch(prefetch)
  return dataset

if __name__ == '__main__':
  import sys
  import os
  import glob
  filenames = glob.glob(os.path.join(sys.argv[1],'*.tfrecord'))
  dataset = build(filenames)
  for feature in dataset:
    speaker_id = feature['speaker_id'].numpy()
    sample_id = feature['sample_id'].numpy()
    filename = feature['filename'].numpy()
    mag = feature['mag'].numpy()
    print(filename)
    for m in mag:
      print(m.shape)
