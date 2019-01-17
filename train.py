from common.config import _ex
from scipy.io import savemat

import os
import glob
import traceback
import torch
import torch.nn as nn
import numpy as np
import random

from common.environment import make_env
from common.model import Model, DistributionalModel, Buffer
from common.misc import to_np, tensor, update_linear_schedule, ppo_loss, dppo_loss, acer_loss, dacer_loss

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


model_dict = {
  'PPO': lambda state_shape, action_shape: Model(state_shape, action_shape),
  'DPPO': lambda state_shape, action_shape: DistributionalModel(state_shape, action_shape),
  'ACER': lambda state_shape, action_shape: Model(state_shape, action_shape),
  'DACER': lambda state_shape, action_shape: DistributionalModel(state_shape, action_shape)
}
loss_dict = {
  'PPO': lambda samples, model: ppo_loss(samples, model),
  'DPPO': lambda samples, model: dppo_loss(samples, model),
  'ACER': lambda samples, model: acer_loss(samples, model),
  'DACER': lambda samples, model: dacer_loss(samples, model)
}



@_ex.capture
def train(model_name, num_processes, max_grad_norm, num_env_steps, log_dir, epoch, env_name, save_dir, use_linear_clip_decay):
  records = []
  envs = [make_env(rank = i) for i in range(num_processes)]
  replaybuffer = Buffer()
  if len(envs) > 1:
    envs = SubprocVecEnv(envs)
  else:
    envs = DummyVecEnv(envs)
  try:
    state_shape = envs.observation_space.shape[0]
    action_shape = envs.action_space.shape[0]
    model = model_dict[model_name](state_shape, action_shape)
    cumpute_loss = loss_dict[model_name]
    optimizer = torch.optim.Adam(model.parameters())
    state = envs.reset()
    returns = 0
    for t in range(num_env_steps//num_processes):
      action, log_prob = model.act(state)
      next_state, reward, done, info = envs.step(to_np(action))
      returns += reward
      replaybuffer.store(zip(state, to_np(action), to_np(log_prob), reward, next_state, 1 - done))
      for i, d in enumerate(done):
        if d:
          records.append((t * num_processes + i, returns[i]))
          if i==0:
            print(returns[0])
          returns[i] = 0
      state = next_state

      if t % 500//num_processes == (500//num_processes-1):
        for _ in range(epoch):
          optimizer.zero_grad()
          loss = cumpute_loss(replaybuffer.sample(), model)
          loss.backward()
          nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
          optimizer.step()
        if model_name == 'PPO' or model_name == 'DPPO':
          replaybuffer.clear()

      if t % (num_env_steps//num_processes//10) == 0:
        i = t//(num_env_steps//num_processes//10)
        torch.save(model.state_dict(), os.path.join(save_dir, model_name,env_name, model_name+str(i)+'.pt'))
      if use_linear_clip_decay:
        update_linear_schedule(optimizer, t * num_processes)
    torch.save(model.state_dict(), os.path.join(save_dir, model_name,env_name, model_name+'_Final.pt'))
    timesteps , sumofrewards = zip(*records)
    savemat(os.path.join(save_dir, model_name,env_name,'returns.mat'),{'timesteps':timesteps, 'returns':sumofrewards})
  except Exception as e:
    traceback.print_exc()
  finally:
    envs.close()
      






@_ex.automain
def run(log_dir, save_dir, model_name, env_name, seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  try:
    os.makedirs(log_dir)
  except OSError:
    files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)
  try:
    os.makedirs(os.path.join(save_dir, model_name,env_name))
  except OSError:
    pass
  train()

