import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

class DownSampleLayer(tf.keras.Model):
  def __init__(self, filters, kernel_size, stride, padding, residual=False):
    super(DownSampleLayer, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding)
    self.inst_norm1 = tfa.layers.InstanceNormalization()
    self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding)
    self.inst_norm2 = tfa.layers.InstanceNormalization()
    self.residual = residual
    if residual:
      self.identity = tf.keras.layers.Conv2D(filters, (1,1), stride, padding)

  def call(self, x):
    h = self.conv1(x)
    h = self.inst_norm1(h)

    gate = self.conv2(x)
    gate = self.inst_norm2(gate)
    
    if self.residual:
      x = self.identity(x)
      return x + h*tf.sigmoid(gate)
    else:
      return h*tf.sigmoid(gate)


class UpSampleLayer(tf.keras.Model):
  def __init__(self, filters, kernel_size, strides, padding, residual=False):
    super(UpSampleLayer, self).__init__()
    self.conv_transpose1 = tf.keras.layers.Conv2DTranspose(
        filters, kernel_size, strides, padding)
    self.inst_norm1 = tfa.layers.InstanceNormalization()
    self.conv_transpose2 = tf.keras.layers.Conv2DTranspose(
        filters, kernel_size, strides, padding)
    self.inst_norm2 = tfa.layers.InstanceNormalization()
    self.residual = residual
    if residual:
      self.identity = tf.keras.layers.Conv2DTranspose(filters, (1,1), strides, padding)


  def call(self, x):
    h = self.conv_transpose1(x)
    h = self.inst_norm1(h)

    gate = self.conv_transpose2(x)
    gate = self.inst_norm2(gate)

    if self.residual:
      x = self.identity(x)
      return x + h*tf.sigmoid(gate)
    else:
      return h*tf.sigmoid(gate)


class AutoEncoder(tf.keras.Model):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.encoder = tf.keras.Sequential([
        DownSampleLayer(32, (9,4), (1,1), 'same'),
        DownSampleLayer(64, (8,4), (2,2), 'same', True),
        DownSampleLayer(128, (8,4), (2,2), 'same', True),
        DownSampleLayer(64, (5,4), (1,1), 'same', True),
        DownSampleLayer(5, (5,10), (1,10), 'same', True)
    ])
    self.decoder = tf.keras.Sequential([
        UpSampleLayer(64, (5,10), (1,10), 'same', True),
        UpSampleLayer(128, (5,4), (1,1), 'same', True),
        UpSampleLayer(64, (8,4), (2,2), 'same', True),
        UpSampleLayer(32, (8,4), (2,2), 'same', True),
        tf.keras.layers.Conv2DTranspose(1, (9,4), (1,1), 'same')
    ])

  def call(self, x, speaker_one_hot):
    encoder_feature = self.encoder(x)
    shape = tf.shape(encoder_feature)
    speaker_one_hot = tf.expand_dims(speaker_one_hot, axis=1)
    speaker_one_hot = tf.expand_dims(speaker_one_hot, axis=1)
    speaker_inform = tf.tile(speaker_one_hot,[1,shape[1],shape[2],1])
    encoder_feature_with_speaker = tf.concat([encoder_feature, speaker_inform], axis=-1)
    decoder_feature = self.decoder(encoder_feature_with_speaker)
    return encoder_feature, decoder_feature


class Discriminator(tf.keras.Model):
  def __init__(self, num_speaker=2):
    super(Discriminator, self).__init__()
    self.conv1 = DownSampleLayer(5, (3,3), (1,1), 'same')
    self.gap = tf.keras.layers.GlobalAvgPool2D()
    self.dense1 = tf.keras.layers.Dense(5, activation=tf.keras.activations.relu)
    self.dense2 = tf.keras.layers.Dense(num_speaker)

  def call(self, latent):
    out = self.conv1(latent)
    out = self.gap(out)
    out = self.dense1(out)
    logits = self.dense2(out)
    return logits

def main():
  mel_feature = np.ones((4, 512, 240, 1), dtype=np.float32)
  speaker_one_hot = np.eye(2)[[0,0,0,0]]
  print(mel_feature.shape)
  # print(mel_feature)
  auto_encoder = AutoEncoder()
  discriminator = Discriminator(2)
  encoded, decoded = auto_encoder(mel_feature, speaker_one_hot)
  logits = discriminator(encoded)
  print(decoded.shape)
  print(logits.shape)
  # print(converted_mel_feature.numpy())
  # print(len(auto_encoder.variables))
  # print(auto_encoder.trainable_weights)

if __name__ == '__main__':
  main()
