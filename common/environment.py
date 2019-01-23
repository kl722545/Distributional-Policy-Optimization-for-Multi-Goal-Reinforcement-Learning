import os
import gym
from baselines import bench
from .config import _ex

@_ex.capture
def make_env(env_name, seed, rank, log_dir):
  def _thunk():
    env = gym.make(env_name)
    if isinstance(env.observation_space,gym.spaces.dict_space.Dict):
      env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
    env.seed(seed + rank)
    if log_dir is not None:
      env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
    return env
  return _thunk

@_ex.capture
def make_goal_env(env_name, seed, rank, log_dir):
  def _thunk():
    env = gym.make(env_name)
    env.seed(seed + rank)
    if log_dir is not None:
      env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
    return env
  return _thunk