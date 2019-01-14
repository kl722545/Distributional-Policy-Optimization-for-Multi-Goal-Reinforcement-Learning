from .config import _ex, DEVICE

import os
import torch
import numpy as np
import torch
import torch.nn as nn


def tensor(x):
  if isinstance(x, torch.Tensor):
    return x
  x = torch.tensor(x, device=DEVICE, dtype=torch.float32)
  return x
def layer_init(layer, gain=np.sqrt(2)):
  nn.init.orthogonal_(layer.weight.data, gain=gain)
  nn.init.constant_(layer.bias.data,0)
  return layer
def to_np(t):
  return t.cpu().detach().numpy() 
@_ex.capture
def update_linear_schedule(optimizer, current_env_steps, num_env_steps, lr):
  """Decreases the learning rate linearly"""
  lr = lr - (lr * (current_env_steps / float(num_env_steps)))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
@_ex.capture    
def cal_target_distri(prob_next, reward, mask, gamma, atoms, delta_atom, categorical_v_min, categorical_v_max):
  atoms_next = reward + gamma * mask * atoms.view(1, -1)
  atoms_next.clamp_(categorical_v_min, categorical_v_max)
  b = (atoms_next - categorical_v_min) / delta_atom
  l = b.floor()
  u = b.ceil()
  d_m_l = (u + (l == u).float() - b) * prob_next
  d_m_u = (b - l) * prob_next
  target_prob = torch.zeros_like(prob_next)
  for i in range(target_prob.size(0)):
      target_prob[i].index_add_(0, l[i].long(), d_m_l[i])
      target_prob[i].index_add_(0, u[i].long(), d_m_u[i])
  return target_prob