from .config import _ex
from torch.distributions import Normal
from .misc import tensor, layer_init
from collections import deque


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
class Model(nn.Module):
  @_ex.capture
  def __init__(self, state_shape, action_shape, hidden_size):
    super(Model, self).__init__()
    self.actor_mean_fc = nn.Sequential(
      layer_init(nn.Linear(state_shape, hidden_size)),
      nn.Tanh(),
      layer_init(nn.Linear(hidden_size, hidden_size)),
      nn.Tanh(),
      layer_init(nn.Linear(hidden_size, action_shape), gain = 1)
      )

    self.actor_logstd = nn.Parameter(torch.zeros(action_shape))
    self.critic_fc = nn.Sequential(
      layer_init(nn.Linear(state_shape, hidden_size)),
      nn.Tanh(),
      layer_init(nn.Linear(hidden_size, hidden_size)),
      nn.Tanh(),
      layer_init(nn.Linear(hidden_size, 1), gain = 1)
      )
  def forward(self, state):
    state = tensor(state)
    actor_mean = self.actor_mean_fc(state)
    value = self.critic_fc(state)
    return Normal(actor_mean, self.actor_logstd.exp()), value
  def act(self, state, deterministic=False):
    state = tensor(state)
    mean = self.actor_mean_fc(state)
    dist = Normal(mean, self.actor_logstd.exp())
    if deterministic:
      action = mean
    else:
      action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob
  def eval(self, state):
    state = tensor(state)
    value = self.critic_fc(state)
    return value

class GoalModel(Model):
  @_ex.capture
  def __init__(self, state_shape, goal_shape, action_shape, hidden_size):
    super(GoalModel, self).__init__(state_shape + goal_shape, action_shape)
  def forward(self, state, goal):
    return super(GoalModel, self).forward(np.concatenate((state, goal), axis=-1))
  def act(self, state, goal, deterministic=False):
    return super(GoalModel, self).act(np.concatenate((state, goal), axis=-1), deterministic)
  def eval(self, state, goal):
    return super(GoalModel, self).eval(np.concatenate((state, goal), axis=-1))

class DistributionalModel(nn.Module):
  @_ex.capture
  def __init__(self, state_shape, action_shape, hidden_size, categorical_v_min, categorical_v_max, categorical_num_atom):
    super(DistributionalModel, self).__init__()
    self.actor_mean_fc = nn.Sequential(
      layer_init(nn.Linear(state_shape, hidden_size)),
      nn.Tanh(),
      layer_init(nn.Linear(hidden_size, hidden_size)),
      nn.Tanh(),
      layer_init(nn.Linear(hidden_size, action_shape), gain = 1)
      )
    self.actor_logstd = nn.Parameter(torch.zeros(action_shape))
    self.critic_logit_fc = nn.Sequential(
      layer_init(nn.Linear(state_shape, hidden_size)),
      nn.Tanh(),
      layer_init(nn.Linear(hidden_size, hidden_size)),
      nn.Tanh(),
      layer_init(nn.Linear(hidden_size, categorical_num_atom), gain = 1)
      )
    self.categorical_v_min = categorical_v_min
    self.categorical_v_max = categorical_v_max
    self.categorical_n_atoms = categorical_num_atom
    self.atoms = tensor(np.linspace(self.categorical_v_min, self.categorical_v_max, self.categorical_n_atoms))
  def forward(self, state):
    state = tensor(state)
    actor_mean = self.actor_mean_fc(state)
    critic_logit = self.critic_logit_fc(state)
    critic_prob = F.softmax(critic_logit, dim=-1)
    critic_log_prob = F.log_softmax(critic_logit, dim=-1)
    return Normal(actor_mean, self.actor_logstd.exp()), critic_prob, critic_log_prob
  def act(self, state, deterministic=False):
    state = tensor(state)
    mean = self.actor_mean_fc(state)
    dist = Normal(mean, self.actor_logstd.exp())
    if deterministic:
      action = dist.mean
    else:
      action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob
  def eval(self, state):
    state = tensor(state)
    logit = self.critic_logit_fc(state)
    prob = F.softmax(logit, dim=-1)
    log_prob = F.log_softmax(logit, dim=-1)
    value = (prob * self.atoms).sum(-1)
    return  value, prob, log_prob

class DistributionalGoalModel(DistributionalModel):
  @_ex.capture
  def __init__(self, state_shape, goal_shape, action_shape, hidden_size, categorical_v_min, categorical_v_max, categorical_num_atom):
    super(DistributionalGoalModel, self).__init__(state_shape + goal_shape, action_shape, hidden_size, categorical_v_min, categorical_v_max, categorical_num_atom)
  def forward(self, state, goal):
    return super(DistributionalGoalModel, self).forward(np.concatenate((state, goal), axis=-1))
  def act(self, state, goal, deterministic=False):
    return super(DistributionalGoalModel, self).act(np.concatenate((state, goal), axis=-1), deterministic)
  def eval(self, state, goal):
    return super(DistributionalGoalModel, self).eval(np.concatenate((state, goal), axis=-1))


class Buffer:
  @_ex.capture
  def __init__(self, num_processes, num_cache_epoch):
    super(Buffer, self).__init__()
    self.storages = [deque([[]],  maxlen=num_cache_epoch+1) for _ in range(num_processes)]
  @_ex.capture
  def store(self, transistions, num_processes):#states,actions,rewards,next_states,masks
    for storage, transistion in zip(self.storages,transistions):
      storage[-1].append(transistion)
      mask = transistion[-1]
      if mask == 0:
        storage.append([])
  @_ex.capture
  def sample(self, num_minibatch, num_step, num_processes):
    for storage in self.storages:
      if len(storage[-1]) == 0:
        storage.pop()
    trajectories = [random.choice(random.choice(self.storages)) for _ in range(num_minibatch)]
    num_step = min(num_step, min(map(lambda trajectory: len(trajectory), trajectories)))
    idx = list(map(lambda trajectory: random.randint(0,max(0, len(trajectory) - num_step)), trajectories))
    minibatch = list(map(lambda trajectory, i: trajectory[i:i+num_step], trajectories, idx))
    for storage in self.storages:
      if storage[-1][-1][-1] == 0:
        storage.append([])    
    return minibatch
  def clear(self):
    for storage in self.storages:
      storage.clear()
      storage.append([])

class GoalBuffer:
  @_ex.capture
  def __init__(self, reward_function, num_processes, num_cache_epoch):
    super(GoalBuffer, self).__init__()
    self.storages = [deque([[]],  maxlen=num_cache_epoch+1) for _ in range(num_processes)]
    self.hindsight_storages = [deque([[]],  maxlen=num_cache_epoch+1) for _ in range(num_processes)]
    self.reward_function = reward_function
  @_ex.capture
  def store(self, transistions, num_processes):
    for storage, hindsight_storage, transistion in zip(self.storages, self.hindsight_storages, transistions):
      storage[-1].append(transistion)
      hindsight_storage[-1].append(copy.deepcopy(transistion))
      mask = transistion[-1]
      if mask == 0:
        storage.append([])
        hindsight_storage.append([])
  @_ex.capture
  def sample(self, num_minibatch, num_step, num_processes):
    for storage, hindsight_storage in zip(self.storages, self.hindsight_storages):
      if len(storage[-1]) == 0:
        storage.pop()
      if len(hindsight_storage[-1]) == 0:
        hindsight_storage.pop()
    tmp_deque = self.storages + self.hindsight_storages
    trajectories = [random.choice(random.choice(tmp_deque)) for _ in range(num_minibatch)]
    num_step = min(num_step, min(map(lambda trajectory: len(trajectory), trajectories)))
    idx = list(map(lambda trajectory: random.randint(0,max(0, len(trajectory) - num_step)), trajectories))
    minibatch = list(map(lambda trajectory, i: trajectory[i:i+num_step], trajectories, idx))
    for storage, hindsight_storage in zip(self.storages, self.hindsight_storages):
      if storage[-1][-1][-1] == 0:
        storage.append([])
      if hindsight_storage[-1][-1][-1] == 0:
        hindsight_storage.append([])            
    return minibatch
  def cal_hindsight_reward(self, achieved_goal, idx):
    hindsight_storage = self.hindsight_storages[idx]
    goal = achieved_goal
    for i in range(len(hindsight_storage[-2])):
      state, desired_goal, achieved_goal, action, log_prob, reward, next_state, mask = hindsight_storage[-2][i]
      desired_goal = goal
      reward = self.reward_function(desired_goal, achieved_goal)
      hindsight_storage[-2][i] = state, desired_goal, achieved_goal, action, log_prob, reward, next_state, mask