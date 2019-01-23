from common.config import _ex
from scipy.io import savemat
from collections import deque

import os
import glob
import traceback
import torch
import torch.nn as nn
import numpy as np
import random
import gym

from common.environment import make_goal_env
from common.model import GoalModel, DistributionalGoalModel, GoalBuffer
from common.misc import to_np, tensor, update_linear_schedule, acher_loss, dacher_loss

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


model_dict = {
  'ACHER': lambda state_shape, goal_shape, action_shape: GoalModel(state_shape, goal_shape, action_shape),
  'DACHER': lambda state_shape, goal_shape, action_shape: DistributionalGoalModel(state_shape, goal_shape, action_shape)
}
loss_dict = {
  'ACHER': lambda samples, model: acher_loss(samples, model),
  'DACHER': lambda samples, model: dacher_loss(samples, model)
}



@_ex.capture
def train(model_name, num_processes, max_grad_norm, num_env_steps, log_dir, epoch, env_name, save_dir, use_linear_clip_decay):
  records = []
  envs = [make_goal_env(rank = i) for i in range(num_processes)]
  replaybuffer = GoalBuffer(lambda desired_goal, achieved_goal :  -10 *((desired_goal-achieved_goal)**2).sum() ** 0.5)
  if len(envs) > 1:
    envs = SubprocVecEnv(envs)
  else:
    envs = DummyVecEnv(envs)
  try:
    state_shape = envs.observation_space.spaces['observation'].shape[0]
    goal_shape = envs.observation_space.spaces['achieved_goal'].shape[0]
    action_shape = envs.action_space.shape[0]
    model = model_dict[model_name](state_shape, goal_shape, action_shape)
    cumpute_loss = loss_dict[model_name]
    optimizer = torch.optim.Adam(model.parameters())
    state_dict = envs.reset()
    returns = 0
    for t in range(num_env_steps//num_processes):
      state = np.array([sd['observation'] for sd in state_dict])
      desired_goal = np.array([sd['desired_goal'] for sd in state_dict])
      achieved_goal = np.array([sd['achieved_goal'] for sd in state_dict])
      action, log_prob = model.act(state, achieved_goal)
      next_state_dict, reward, done, info = envs.step(to_np(action))
      next_state = np.array([sd['observation'] for sd in next_state_dict])
      returns += reward
      replaybuffer.store(list(zip(state, desired_goal, achieved_goal, to_np(action), to_np(log_prob), reward, next_state, 1 - done)))
      for i, d in enumerate(done):
        if d:
          records.append((t * num_processes + i, returns[i]))
          goal = achieved_goal[i]
          replaybuffer.cal_hindsight_reward(goal, i)
          if i==0:
            print(returns[0])
          returns[i] = 0
      state_dict = next_state_dict
      if t % 500//num_processes == (500//num_processes-1):
        for _ in range(epoch):
          optimizer.zero_grad()
          loss = cumpute_loss(replaybuffer.sample(), model)
          loss.backward()
          nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
          optimizer.step()

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

