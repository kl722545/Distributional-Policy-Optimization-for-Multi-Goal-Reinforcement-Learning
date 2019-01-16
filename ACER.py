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
from common.model import Model, Buffer
from common.misc import to_np, update_linear_schedule, tensor

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

@_ex.capture
def cumpute_loss(samples, model, gamma, tau, clip_param, value_loss_coef, trace_clip_max_c, trace_clip_max_rho, use_reward_clip):
  i = 0
  gaes = []
  ratios = []
  actor_loss = 0
  critic_loss = 0
  for batch_tran in reversed(list(zip(*samples))):
    state, action, old_log_prob,  reward, next_state, mask = map(lambda item:tensor(item), zip(*batch_tran))
    if use_reward_clip:
      reward = torch.clamp(reward, -1, 1)
    reward = reward.unsqueeze(dim=-1)
    mask = mask.unsqueeze(dim=-1)
    new_distri, value = model(state)  
    new_log_prob = new_distri.log_prob(action)
    ratio = (new_log_prob - old_log_prob).exp().sum(-1).unsqueeze(dim=-1)
    clip_ratio_c = torch.clamp(ratio.detach(), 0, trace_clip_max_c)
    clip_ratio_rho = torch.clamp(ratio.detach(), 0, trace_clip_max_rho)
    delta =  clip_ratio_rho * (reward + gamma * mask * model.eval(next_state).detach() - value)
    gaes = gaes + [delta + gamma  * tau * clip_ratio_c  * (gaes[-1] if len(gaes)!= 0 else 0)]
    critic_loss += (delta**2).mean()
    ratios = ratios + [ratio]
    i += 1

  gaes = torch.cat(gaes,dim=-1).detach()
  gaes = gaes - gaes.mean(dim=-1).unsqueeze(dim=-1)
  gaes = gaes / (((gaes * gaes).mean(dim=-1)+ 1e-8).sqrt().unsqueeze(dim=-1) )
  gaes = gaes.split(1, dim=-1)
  for gae,ratio in zip(gaes,ratios):
    actor_loss +=  - torch.min(ratio* gae, torch.clamp(ratio, 1-clip_param , 1+clip_param) * gae).mean()
  loss = actor_loss + value_loss_coef * critic_loss
  assert not torch.isnan(loss)
  loss /= i
  return loss

@_ex.capture
def train(num_processes, max_grad_norm, num_env_steps, log_dir, epoch, env_name, save_dir, use_linear_clip_decay):
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
    model = Model(state_shape, action_shape)
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
      if t % (num_env_steps//num_processes//10) == 0:
        i = t//(num_env_steps//num_processes//10)
        torch.save(model.state_dict(), os.path.join(save_dir, 'ACER',env_name, 'ACER'+str(i)+'.pt'))
      if use_linear_clip_decay:
        update_linear_schedule(optimizer, t * num_processes)
    torch.save(model.state_dict(), os.path.join(save_dir, 'ACER',env_name,'ACER_Final.pt'))
    timesteps , sumofrewards = zip(*records)
    savemat(os.path.join(save_dir, 'ACER',env_name,'returns.mat'),{'timesteps':timesteps, 'returns':sumofrewards})
  except Exception as e:
    traceback.print_exc()
  finally:
    envs.close()
      






@_ex.automain
def run(log_dir, save_dir, env_name, seed):
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
    os.makedirs(os.path.join(save_dir, 'ACER',env_name))
  except OSError:
    pass
  train()

