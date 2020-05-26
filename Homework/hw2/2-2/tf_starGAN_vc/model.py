import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

class DownSampleLayer(tf.keras.Model):
  def __init__(self, filters, kernel_size, stride, padding):
    super(DownSampleLayer, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding)
    self.inst_norm1 = tfa.layers.InstanceNormalization()

    self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding)
    self.inst_norm2 = tfa.layers.InstanceNormalization()

  def call(self, x, training):
    h = self.conv1(x)
    h = self.inst_norm1(h, training=training)

    gate = self.conv2(x)
    gate = self.inst_norm2(gate, training=training)

    return h * tf.sigmoid(gate)


class UpSampleLayer(tf.keras.Model):
  def __init__(self, filters, kernel_size, strides, padding):
    super(UpSampleLayer, self).__init__()
    self.conv_transpose1 = tf.keras.layers.Conv2DTranspose(
        filters, kernel_size, strides, padding)
    self.inst_norm1 = tfa.layers.InstanceNormalization()

    self.conv_transpose2 = tf.keras.layers.Conv2DTranspose(
        filters, kernel_size, strides, padding)
    self.inst_norm2 = tfa.layers.InstanceNormalization()


  def call(self, x, training):
    h = self.conv_transpose1(x)
    h = self.inst_norm1(h, training=training)

    gate = self.conv_transpose2(x)
    gate = self.inst_norm2(gate, training=training)

    return h * tf.sigmoid(gate)


class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()
    self.down_layers = [
        DownSampleLayer(32, (3,9), (1,1), 'same'),
        DownSampleLayer(64, (4,8), (2,2), 'same'),
        DownSampleLayer(128, (4,8), (2,2), 'same'),
        DownSampleLayer(64, (3,5), (1,1), 'same'),
        DownSampleLayer(5, (9,5), (9,1), 'same')]
    self.up_layers = [
        UpSampleLayer(64, (9,5), (9,1), 'same'),
        UpSampleLayer(128, (3,5), (1,1), 'same'),
        UpSampleLayer(64, (4,8), (2,2), 'same'),
        UpSampleLayer(32, (4,8), (2,2), 'same')]
    self.deconv = tf.keras.layers.Conv2DTranspose(1, (3,9), (1,1), 'same')

  def call(self, x, c, training):
    out = x
    for down_layer in self.down_layers:
      out = down_layer(out, training)
    c = tf.reshape(c, (tf.shape(c)[0], 1, 1, tf.shape(c)[1]))

    for up_layer in self.up_layers:
      c_tile = tf.tile(c, (1,tf.shape(out)[1], tf.shape(out)[2],1))
      out = tf.concat([out, c_tile], -1)
      out = up_layer(out, training)
    out = self.deconv(out)

    return out


class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.down_layers = [
        DownSampleLayer(32, (3,9), (1,1), 'same'),
        DownSampleLayer(32, (3,8), (1,2), 'same'),
        DownSampleLayer(32, (3,8), (1,2), 'same'),
        DownSampleLayer(32, (3,6), (1,2), 'same')
    ]
    self.conv = tf.keras.layers.Conv2D(1, (36,5), (36,1), 'same')
    self.pool = tf.keras.layers.GlobalAveragePooling2D()

  def call(self, x, c, training):
    out = x
    c = tf.reshape(c, (tf.shape(c)[0], 1, 1, tf.shape(c)[1]))
    for down_layer in self.down_layers:
      c_tile = tf.tile(c, (1,tf.shape(out)[1], tf.shape(out)[2],1))
      out = tf.concat([out, c_tile], -1)
      out = down_layer(out, training)
    c_tile = tf.tile(c, (1,tf.shape(out)[1], tf.shape(out)[2],1))
    out = tf.concat([out, c_tile], -1)
    out = self.conv(out)
    out = self.pool(out)
    # out = tf.nn.sigmoid(out)
    return out


class DomainClassifier(tf.keras.Model):
  def __init__(self, speakers=2):
    super(DomainClassifier, self).__init__()
    self.down_layers = [
        DownSampleLayer(8, (4,4), (2,2), 'same'),
        DownSampleLayer(16, (4,4), (2,2), 'same'),
        DownSampleLayer(32, (4,4), (2,2), 'same'),
        DownSampleLayer(16, (3,4), (1,2), 'same')]
    self.conv = tf.keras.layers.Conv2D(speakers, (1,4), (1,2), 'same')
    self.pool = tf.keras.layers.GlobalAveragePooling2D()

  def call(self, x, training):
    out = x[:,:8,:,:]
    for down_layer in self.down_layers:
      out = down_layer(out)
    out = self.conv(out)
    out = self.pool(out)
    out = tf.nn.softmax(out)
    return out

def main():
  mel_feature = np.ones((1, 36, 512, 1), dtype=np.float32)
  speaker_one_hot = np.asarray([[0,1,0,0]], dtype=np.float32)
  print(mel_feature.shape)
  G = Generator()
  fake_G = G(mel_feature, speaker_one_hot, True)
  D = Discriminator()
  fake_D = D(fake_G, speaker_one_hot, True)
  C = DomainClassifier()
  fake_C = C(fake_G, True)
  print(fake_G.shape)
  print(fake_D.shape)
  print(fake_C.shape)

if __name__ == '__main__':
  main()
