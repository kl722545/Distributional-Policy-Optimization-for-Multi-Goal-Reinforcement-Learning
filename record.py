from common.config import _ex
from scipy.io import savemat

import os
import glob
import traceback
import torch
import torch.nn as nn
import numpy as np
import random

from common.misc import to_np
from common.environment import make_env
from common.model import Model, DistributionalModel
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

model_dict = {
  'PPO': lambda state_shape, action_shape: Model(state_shape, action_shape),
  'DPPO': lambda state_shape, action_shape: DistributionalModel(state_shape, action_shape),
  'ACER': lambda state_shape, action_shape: Model(state_shape, action_shape),
  'DACER': lambda state_shape, action_shape: DistributionalModel(state_shape, action_shape)
}

@_ex.capture
def record(num_env_steps, num_processes, log_dir, env_name, model_name, save_dir, categorical_v_min, categorical_v_max, categorical_num_atom):
  records = []
  epoch = 0
  env = make_env(rank = 0)()
  try:
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]
    model = model_dict[model_name](state_shape, action_shape)
    for i in range(10):
      print('recording '+ model_name+str(i)+'...')
      state_dict = torch.load(os.path.join(save_dir, model_name,env_name, model_name+str(i)+'.pt'))
      model.load_state_dict(state_dict)
      done = False
      state = env.reset()
      records = []
      returns = 0
      while not done:
        observation = env.render(mode='rgb_array')
        action, _ = model.act(state)
        action_dist, value_dist, _ = model(state)
        next_state, reward, done, info = env.step(to_np(action))
        records.append((observation, to_np(action_dist.mean), to_np(action_dist.stddev), to_np(action), to_np(value_dist), reward))
        state = next_state
        returns += reward
      print("return : {0}".format(returns))
      observations, means, stddevs, actions, value_dists, rewards = zip(*records)
      savemat(os.path.join(save_dir, model_name,env_name,model_name+str(i)+'.mat'),
        {'observations':observations,
        'means':means,
        'stddevs':stddevs,
        'actions':actions,
        'value_dists':value_dists,
        'rewards':rewards,
        'categorical_v_min':categorical_v_min,
        'categorical_v_max':categorical_v_max,
        'categorical_num_atom':categorical_num_atom
        })
    print('recording '+ model_name+'_Final'+'...')
    state_dict = torch.load(os.path.join(save_dir, model_name,env_name, model_name+'_Final.pt'))
    model.load_state_dict(state_dict) 
    done = False
    state = env.reset()
    records = []
    returns = 0
    while not done:
      observation = env.render(mode='rgb_array')
      action, log_prob = model.act(state)
      action_dist, value_dist, _ = model(state)
      next_state, reward, done, info = env.step(to_np(action))
      records.append((observation, to_np(action_dist.mean), to_np(action_dist.stddev), to_np(action), to_np(value_dist), reward))
      state = next_state
      returns += reward
    print("return : {0}".format(returns))
    observations, means, stddevs, actions, value_dists, rewards = zip(*records)
    savemat(os.path.join(save_dir, model_name,env_name,model_name+'_Final.mat'),
      {'observations':observations,
      'means':means,
      'stddevs':stddevs,
      'actions':actions,
      'value_dists':value_dists,
      'rewards':rewards,
      'categorical_v_min':categorical_v_min,
      'categorical_v_max':categorical_v_max,
      'categorical_num_atom':categorical_num_atom
      })
  except Exception as e:
    traceback.print_exc()
  finally:
    env.close()



@_ex.automain
def run(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  record()