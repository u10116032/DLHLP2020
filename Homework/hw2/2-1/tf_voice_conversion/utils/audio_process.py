import librosa
import numpy as np

sampling_rate = 16000 # Sampling rate.
n_fft = 1024 # fft points (samples)
frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds
hop_length = int(sampling_rate * frame_shift) # samples
win_length = int(sampling_rate * frame_length) # samples
n_mels = 80 # Number of Mel banks to generate
power = 1.2 # Exponent for amplifying the predicted magnitude
n_iter = 200 # Number of inversion iterations
use_log_magnitude = True # if False, use magnitude


def wav2spectrogram(filepath):
  '''
  Generate melspectogram of each wav.
  If we do further log and DFT, we can get MFCC.
  return:
    (mag, mel)
      mag: (T, 1+n_fft//2)
      mel: (T, n_mels)
  '''

  y, sr = librosa.load(filepath, sr=sampling_rate)
  y, _ = librosa.effects.trim(y)

  # stft. D: (1+n_fft//2, T)
  D = librosa.stft(y=y,
                   n_fft=n_fft,
                   hop_length=hop_length,
                   win_length=win_length)

  mag = np.abs(D)
  power = mag**2

  # mel spectrogram, mel: (n_mels, T)
  mel = librosa.feature.melspectrogram(S=power, n_mels=n_mels)

  mag = np.transpose(mag.astype(np.float32))
  mel = np.transpose(mel.astype(np.float32))

  return (mag, mel)


def spectrogram2wav():
  pass

if __name__ == '__main__':
  import sys
  mag, mel = wav2spectrogram(sys.argv[1])
  print(mag.shape)
  print(mel.shape)
