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
def test(num_env_steps, num_processes, log_dir, env_name, model_name, save_dir):
  records = []
  epoch = 0
  envs = [make_env(rank = i) for i in range(num_processes)]
  if len(envs) > 1:
    envs = SubprocVecEnv(envs)
  else:
    envs = DummyVecEnv(envs)
  try:
    state_shape = envs.observation_space.shape[0]
    action_shape = envs.action_space.shape[0]
    model = model_dict[model_name](state_shape, action_shape)
    state_dict = torch.load(os.path.join(save_dir, model_name,env_name, model_name+'_Final.pt'))
    model.load_state_dict(state_dict)
    state = envs.reset()
    returns = 0
    for t in range(num_env_steps//num_processes):
      action, log_prob = model.act(state)
      next_state, reward, done, info = envs.step(to_np(action))
      returns += reward
      for i, d in enumerate(done):
        if d:
          records.append(returns[i])
          returns[i] = 0
          epoch += 1
      if epoch >= 100:
        break
      state = next_state
    records = np.array(records)
    print("# of epoch: {0}".format(epoch))
    print("mean: {0}".format(np.mean(records)))
    print("std: {0}".format(np.std(records)))
    print("max: {0}".format(np.max(records)))
    print("min: {0}".format(np.min(records)))
    print("median: {0}".format(np.median(records)))
  except Exception as e:
    traceback.print_exc()
  finally:
    envs.close()



@_ex.automain
def run(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  test()