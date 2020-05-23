import numpy as np
import tensorflow as tf
import pickle

class McepNormalizer():

  def __init__(self, norm_dict_pkl_path):
    with open(norm_dict_pkl_path, 'rb') as file:
      self.norm_dict = pickle.load(file)
 
  def mcep_norm(self, mcep, speaker):
    mean = self.norm_dict[speaker]['mcep_mean']
    std = self.norm_dict[speaker]['mcep_std']
    mean = np.reshape(mean,[-1,1])
    std = np.reshape(std,[-1,1])
    
    return (mcep - mean) / std

  def batch_mcep_norm(self, batch_mcep, batch_speaker, is_tensor=True):
    out_list = []
    for idx in range(batch_mcep.shape[0]):
      mcep = batch_mcep[idx,:,:]
      speaker = str(int(batch_speaker[idx])) 
      normed_mcep = self.mcep_norm(mcep, speaker)
      out_list.append(normed_mcep)
    if is_tensor:
      out = tf.stack(out_list, axis=0)
    else:
      out = np.stack(out_list, axis=0)
    return out
    
  
  def mcep_denorm(self, mcep_norm, speaker):
    mean = self.norm_dict[speaker]['mcep_mean']
    std = self.norm_dict[speaker]['mcep_std']
    mean = np.reshape(mean,[-1,1])
    std = np.reshape(std,[-1,1])
    return mcep_norm * std + mean

  def f0_convert(self, f0, src_speaker, target_speaker):
    print('src:', src_speaker, 'tgr: ', target_speaker)
    src_mean = self.norm_dict[src_speaker]['log_f0_mean']
    src_std = self.norm_dict[src_speaker]['log_f0_std']
    target_mean = self.norm_dict[target_speaker]['log_f0_mean']
    target_std = self.norm_dict[target_speaker]['log_f0_std']
    
    f0_converted = np.exp((np.log(f0+1e-21) - src_mean) / src_std * target_std + target_mean)
    return f0_converted
