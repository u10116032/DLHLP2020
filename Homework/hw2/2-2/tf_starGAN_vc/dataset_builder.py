import tensorflow as tf

def build(filepaths, batch=16, prefetch=4, epoch=-1):
  feature_desciption = {
    'speaker_id': tf.io.FixedLenFeature([], tf.int64),
    'sample_id': tf.io.FixedLenFeature([], tf.int64),
    'filename': tf.io.FixedLenFeature([], tf.string),
    'mel': tf.io.FixedLenFeature([], tf.string),
    'mel_shape': tf.io.FixedLenFeature([2,], tf.int64),
    'mag': tf.io.FixedLenFeature([], tf.string),
    'mag_shape': tf.io.FixedLenFeature([2,], tf.int64),
    'mfcc': tf.io.FixedLenFeature([], tf.string),
    'mfcc_shape': tf.io.FixedLenFeature([2,], tf.int64),
    'mcep': tf.io.FixedLenFeature([], tf.string),
    'mcep_shape': tf.io.FixedLenFeature([2,], tf.int64)
  }
  def _parse_feature(example_proto):
    feature = tf.io.parse_single_example(example_proto, feature_desciption)
    speaker_id = feature['speaker_id']
    sample_id = feature['sample_id']
    filename = feature['filename']
    mag_shape = feature['mag_shape']
    mag = tf.io.decode_raw(feature['mag'], tf.float32)
    mag = tf.reshape(mag, mag_shape)
    pad_size = 4 - mag_shape[1]%4
    mag = tf.pad(mag, [[0,0],[0,pad_size]])
    mel_shape = feature['mel_shape']
    mel = tf.io.decode_raw(feature['mel'], tf.float32)
    mel = tf.reshape(mel, mel_shape)
    mel = tf.pad(mel, [[0,0],[0,pad_size]])
    mfcc_shape = feature['mfcc_shape']
    mfcc = tf.io.decode_raw(feature['mfcc'], tf.float32)
    mfcc = tf.reshape(mfcc, mfcc_shape)
    mfcc = tf.pad(mfcc, [[0,0],[0,pad_size]])
    mcep_shape = feature['mcep_shape']
    mcep = tf.io.decode_raw(feature['mcep'], tf.float64)
    mcep = tf.reshape(mcep, mcep_shape)
    mcep = tf.pad(mcep, [[0,0],[0,pad_size]])

    mag = tf.slice(mag, [0,0],
        [tf.shape(mag)[0], tf.math.minimum(tf.shape(mag)[1], 512)])
    mel = tf.slice(mel, [0,0],
        [tf.shape(mel)[0], tf.math.minimum(tf.shape(mel)[1], 512), ])
    mfcc = tf.slice(mfcc, [0,0],
        [tf.shape(mfcc)[0], tf.math.minimum(tf.shape(mfcc)[1], 512), ])
    mcep = tf.slice(mcep, [0,0],
        [tf.shape(mcep)[0], tf.math.minimum(tf.shape(mcep)[1], 512), ])

    tensor_dict = {}
    tensor_dict['speaker_id'] = speaker_id
    tensor_dict['sample_id'] = sample_id
    tensor_dict['filename'] = filename
    tensor_dict['mag'] = mag
    tensor_dict['mel'] = mel
    tensor_dict['mfcc'] = mfcc
    tensor_dict['mcep'] = mcep
    return tensor_dict

  pad_shape = {}
  pad_shape['speaker_id'] = []
  pad_shape['sample_id'] = []
  pad_shape['filename'] = []
  pad_shape['mag'] = [None, None]
  pad_shape['mel'] = [None, None]
  pad_shape['mfcc'] = [None, None]
  pad_shape['mcep'] = [None, None]

  dataset = tf.data.Dataset.from_tensor_slices(filepaths)
  dataset = dataset.interleave(lambda x:
      tf.data.TFRecordDataset(x).map(_parse_feature,
      num_parallel_calls=len(filepaths)),
      cycle_length=8, block_length=8)
  dataset = dataset.repeat(epoch)
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
    mcep = feature['mcep'].numpy()
    print(speaker_id.shape)
    for m in mcep:
      print(m.shape)
