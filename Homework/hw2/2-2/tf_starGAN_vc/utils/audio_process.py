import librosa
import numpy as np
import os
import pyworld

sampling_rate = 22050 # Sampling rate.
n_fft = 2048 # fft points (samples)
frame_shift = 0.005 # seconds
frame_length = 0.05 # seconds
hop_length = int(sampling_rate * frame_shift) # samples
win_length = int(sampling_rate * frame_length) # samples
n_mels = 36 # Number of Mel banks to generate
use_log_magnitude = True # if False, use magnitude
preemphasis = 0.97


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
  # y = np.append(y[0], y[1:]-preemphasis*y[:-1])

  # stft. D: (1+n_fft//2, T)
  D = librosa.stft( y=y,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length )

  mag = np.abs(D)
  power = mag**2

  # mel spectrogram, mel: (n_mels, T)
  mel = librosa.feature.melspectrogram(S=power, n_mels=n_mels)

  return (mag, mel)


def melspectrogram2wav(mel):
  '''
  Invert mel to wav
  args:
    mel: shape (n_mels, T)
  return:
    wav: shape(T*sampling_rate*hop_length,) or ((T-1)*sampling_rate*hop_length,)
  '''
  return librosa.feature.inverse.mel_to_audio( mel,
                                               sr=sampling_rate,
                                               n_fft=n_fft,
                                               hop_length=hop_length,
                                               win_length=win_length )


def wav2mfcc(filepath):
  '''
  Generate MFCC in mono.
  return:
    mfcc with shape (n_mfcc, T) and type np.float
  '''
  y, sr = librosa.load(filepath, sr=sampling_rate, mono=True)
  y, _ = librosa.effects.trim(y)
  # y = np.append(y[0], y[1:]-preemphasis*y[:-1])
  mfcc = librosa.feature.mfcc(y=y, sr=sampling_rate, n_mfcc=n_mels,
      hop_length=hop_length, n_fft=n_fft)
  return mfcc


def mfcc2wav(mfcc):
  '''
  Invert mfcc to wav
  args:
    mel: shape (n_mels, T)
  return:
    wav: shape (T*sampling_rate + T*hop_length,)
  '''
  wav = librosa.feature.inverse.mfcc_to_audio(mfcc, sr=sampling_rate,
      n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
  return wav


def write_wav(wav, filename=None):
  '''
  dump wav file
  args:
    wav: np array with shape (samplings, )
    filename: dump path
  return:
    absolute file path
  '''
  if filename is None:
    filename = 'reconstructed.wav'
  librosa.output.write_wav(filename, wav, sampling_rate)
  return os.path.abspath(filename)


def wav2mcep(filepath):
    '''
    cal mcep given wav singnal

    return:
      f0: shape [ T, ]
      ap: shape [ T, sampling_rate/2 + 1 ]
      sp: shape [ T, sampling_rate/2 + 1 ]
      coded_sp: shape (n_mels x T)
    '''
    y, sr = librosa.load(filepath, sr=sampling_rate)
    y, _ = librosa.effects.trim(y)
    y = np.asarray(y, dtype=np.double)

    f0, timeaxis = pyworld.harvest(y, sr)
    sp = pyworld.cheaptrick(y, f0, timeaxis, sampling_rate, fft_size=n_fft)
    ap = pyworld.d4c(y, f0, timeaxis, sampling_rate, fft_size=n_fft)
    mcep = pyworld.code_spectral_envelope(sp, sampling_rate, n_mels)

    mcep = mcep.T # dim x n

    return f0, ap, sp, mcep

def mcep2wav(mcep, f0, ap):
  decoded_sp = pyworld.decode_spectral_envelope(mcep, sampling_rate,
      fft_size=n_fft)
  wav = pyworld.synthesize(f0, decoded_sp, ap, sampling_rate)
  return wav


if __name__ == '__main__':
  import sys
  # mag, mel = wav2spectrogram(sys.argv[1])
  # print('mag.shape: ', mag.shape)
  # print('mel.shape: ', mel.shape)
  # reconstructed_wav = melspectrogram2wav(mel)
  # print(reconstructed_wav.shape)
  # mfcc = wav2mfcc(sys.argv[1])
  # reconstructed_wav = mfcc2wav(mfcc)
  # print(write_wav(reconstructed_wav))
  # mfcc = wav2mfcc(sys.argv[1])
  # print(mfcc.shape)
  f0, ap, sp, mcep = wav2mcep(sys.argv[1])
  reconstructed_wav = mcep2wav(mcep.T, f0, ap)
  print(f0.shape)
  print(ap.shape)
  print(sp.shape)
  print(mcep.shape)
  print(write_wav(reconstructed_wav))
