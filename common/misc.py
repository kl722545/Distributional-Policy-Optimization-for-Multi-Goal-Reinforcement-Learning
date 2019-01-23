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
def cal_target_distri(prob_next, reward, mask, gamma, atoms, categorical_num_atom, categorical_v_min, categorical_v_max):
  atoms_next = reward + gamma * mask * atoms.view(1, -1)
  atoms_next.clamp_(categorical_v_min, categorical_v_max)
  delta_atom = (categorical_v_max - categorical_v_min) / float(categorical_num_atom - 1)
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

@_ex.capture
def ppo_loss(samples, model, gamma, tau, clip_param, value_loss_coef):
  i = 0
  gaes = []
  ratios = []
  actor_loss = 0
  critic_loss = 0
  for batch_tran in reversed(list(zip(*samples))):
    state, action, old_log_prob,  reward, next_state, mask = map(lambda item:tensor(item), zip(*batch_tran))
    reward = reward.unsqueeze(dim=-1)
    mask = mask.unsqueeze(dim=-1)
    new_distri, value = model(state)  
    new_log_prob = new_distri.log_prob(action)
    delta =  reward + gamma * mask * model.eval(next_state).detach() - value
    gaes = gaes + [delta + gamma * tau  * (gaes[-1] if len(gaes)!= 0 else 0)]
    critic_loss += (delta**2).mean()
    ratios = ratios + [(new_log_prob - old_log_prob).exp().sum(-1).unsqueeze(dim=-1)]
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
def dppo_loss(samples, model, gamma, tau, clip_param, value_loss_coef):
  i = 0
  gaes = []
  ratios = []
  actor_loss = 0
  critic_loss = 0
  for batch_tran in reversed(list(zip(*samples))):
    state, action, old_log_prob,  reward, next_state, mask = map(lambda item:tensor(item), zip(*batch_tran))
    reward = reward.unsqueeze(dim=-1)
    mask = mask.unsqueeze(dim=-1)
    new_distri, value_distri, log_value_distri = model(state)  
    new_log_prob = new_distri.log_prob(action)
    next_value, next_value_distri, _ = model.eval(next_state)
    value , _, _ = model.eval(state)
    target_value_distri = cal_target_distri(next_value_distri.detach(), reward, mask, atoms = model.atoms)
    distance =  - (target_value_distri * (log_value_distri- (target_value_distri+1e-8).log())).sum(-1)
    delta =  reward + gamma * mask * next_value.detach() - value.detach()
    delta = torch.sign(delta) * distance.detach()
    gaes = gaes + [delta + gamma * tau  * (gaes[-1] if len(gaes)!= 0 else 0)]
    critic_loss += distance.mean()
    ratios = ratios + [(new_log_prob - old_log_prob).exp().sum(-1).unsqueeze(dim=-1)]
    i += 1
  for gae,ratio in zip(gaes,ratios):
    actor_loss +=  - torch.min(ratio* gae, torch.clamp(ratio, 1-clip_param , 1+clip_param) * gae).mean()
  loss = actor_loss + value_loss_coef * critic_loss
  assert not torch.isnan(loss)
  loss /= i
  return loss

@_ex.capture
def acer_loss(samples, model, gamma, tau, clip_param, value_loss_coef, trace_clip_max_c, trace_clip_max_rho, use_reward_clip):
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
def dacer_loss(samples, model, gamma, tau, clip_param, value_loss_coef, trace_clip_max_c, trace_clip_max_rho, use_reward_clip):
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
    new_distri, value_distri, log_value_distri = model(state)  
    new_log_prob = new_distri.log_prob(action)
    ratio = (new_log_prob - old_log_prob).exp().sum(-1).unsqueeze(dim=-1)
    clip_ratio_c = torch.clamp(ratio.detach(), 0, trace_clip_max_c)
    clip_ratio_rho = torch.clamp(ratio.detach(), 0, trace_clip_max_rho)
    next_value, next_value_distri, _ = model.eval(next_state)
    value , _, _ = model.eval(state)
    target_value_distri = cal_target_distri(next_value_distri.detach(), reward, mask, atoms = model.atoms)
    distance =  - clip_ratio_rho * (target_value_distri * (log_value_distri- (target_value_distri+1e-8).log())).sum(-1)
    delta =  reward + gamma * mask * next_value.detach() - value.detach()
    delta = torch.sign(delta) * distance.detach()
    gaes = gaes + [delta + gamma  * tau * clip_ratio_c  * (gaes[-1] if len(gaes)!= 0 else 0)]
    critic_loss += distance.mean()
    ratios = ratios + [ratio]
    i += 1
  for gae,ratio in zip(gaes,ratios):
    actor_loss +=  - torch.min(ratio* gae, torch.clamp(ratio, 1-clip_param , 1+clip_param) * gae).mean()
  loss = actor_loss + value_loss_coef * critic_loss
  assert not torch.isnan(loss)
  loss /= i
  return loss

@_ex.capture
def acher_loss(samples, model, gamma, tau, clip_param, value_loss_coef, trace_clip_max_c, trace_clip_max_rho, use_reward_clip):
  i = 0
  gaes = []
  ratios = []
  actor_loss = 0
  critic_loss = 0
  for batch_tran in reversed(list(zip(*samples))):
    state, desired_goal, achieved_goal, action, old_log_prob, reward, next_state, mask = map(lambda item:tensor(item), zip(*batch_tran))
    if use_reward_clip:
      reward = torch.clamp(reward, -1, 1)
    reward = reward.unsqueeze(dim=-1)
    mask = mask.unsqueeze(dim=-1)
    new_distri, value = model(state, desired_goal)  
    new_log_prob = new_distri.log_prob(action)
    ratio = (new_log_prob - old_log_prob).exp().sum(-1).unsqueeze(dim=-1)
    clip_ratio_c = torch.clamp(ratio.detach(), 0, trace_clip_max_c)
    clip_ratio_rho = torch.clamp(ratio.detach(), 0, trace_clip_max_rho)
    delta =  clip_ratio_rho * (reward + gamma * mask * model.eval(next_state, desired_goal).detach() - value)
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
def dacher_loss(samples, model, gamma, tau, clip_param, value_loss_coef, trace_clip_max_c, trace_clip_max_rho, use_reward_clip):
  i = 0
  gaes = []
  ratios = []
  actor_loss = 0
  critic_loss = 0
  for batch_tran in reversed(list(zip(*samples))):
    state, desired_goal, achieved_goal, action, old_log_prob, reward, next_state, mask = map(lambda item:tensor(item), zip(*batch_tran))
    if use_reward_clip:
      reward = torch.clamp(reward, -1, 1)
    reward = reward.unsqueeze(dim=-1)
    mask = mask.unsqueeze(dim=-1)
    new_distri, value_distri, log_value_distri = model(state, desired_goal)  
    new_log_prob = new_distri.log_prob(action)
    ratio = (new_log_prob - old_log_prob).exp().sum(-1).unsqueeze(dim=-1)
    clip_ratio_c = torch.clamp(ratio.detach(), 0, trace_clip_max_c)
    clip_ratio_rho = torch.clamp(ratio.detach(), 0, trace_clip_max_rho)
    next_value, next_value_distri, _ = model.eval(next_state, desired_goal)
    value , _, _ = model.eval(state)
    target_value_distri = cal_target_distri(next_value_distri.detach(), reward, mask, atoms = model.atoms)
    distance =  - clip_ratio_rho * (target_value_distri * (log_value_distri- (target_value_distri+1e-8).log())).sum(-1)
    delta =  reward + gamma * mask * next_value.detach() - value.detach()
    delta = torch.sign(delta) * distance.detach()
    gaes = gaes + [delta + gamma  * tau * clip_ratio_c  * (gaes[-1] if len(gaes)!= 0 else 0)]
    critic_loss += distance.mean()
    ratios = ratios + [ratio]
    i += 1
  for gae,ratio in zip(gaes,ratios):
    actor_loss +=  - torch.min(ratio* gae, torch.clamp(ratio, 1-clip_param , 1+clip_param) * gae).mean()
  loss = actor_loss + value_loss_coef * critic_loss
  assert not torch.isnan(loss)
  loss /= i
  return loss