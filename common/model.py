from .config import _ex
from torch.distributions import Normal
from .misc import tensor, layer_init
from collections import deque


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

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
  def sample(self, num_minibatch, num_step, num_processes, num_cache_epoch):
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
