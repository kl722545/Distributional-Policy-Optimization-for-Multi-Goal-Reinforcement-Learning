import torch
import numpy as np

from sacred import Experiment

DEVICE = torch.device('cpu')
_ex = Experiment()

@_ex.config
def _config():
  hidden_size = 64
  lr = 3e-4 #learning rate
  gamma = 0.995 #discount factor for rewards
  value_loss_coef = 1 #value loss coefficient
  max_grad_norm = 0.5 #max value of gradients
  cuda_deterministic = False #determinism when using CUDA
  num_processes = 10 #training CPU processes to use
  num_env_steps = int(2e6) #number of environment steps to train
  env_name = 'InvertedPendulum-v2' #environment to train on
  log_dir = './log' #directory to save agent logs
  save_dir = './trained_model' #directory to save agent models
  use_cuda = False and torch.cuda.is_available() #CUDA training
  use_linear_clip_decay = True #use a linear schedule on the PPO clipping parameter
  #GAE
  tau = 0.95 #gae parameter
  num_step = 2048 #number of forward steps
  num_cache_epoch = 10 #number of cache epoch for each process
  #PPO
  epoch = 10 #number of PPO epochs 
  num_minibatch = 32 #number of batches for PPO
  clip_param = 0.2 
  #Distribution value net
  categorical_v_min = -200
  categorical_v_max = 200
  categorical_num_atom = 101
  atoms = torch.linspace(categorical_v_min, categorical_v_max, categorical_num_atom, device=DEVICE, dtype=torch.float32)
  delta_atom = (categorical_v_max - categorical_v_min) / float(categorical_num_atom - 1)
  #V-trace
  trace_clip_max_rho = 10
  trace_clip_max_c = 10