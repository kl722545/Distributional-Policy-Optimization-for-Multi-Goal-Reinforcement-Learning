import os
import gym
from baselines import bench
from .config import _ex

@_ex.capture
def make_env(env_name, _seed, rank, log_dir):
  def _thunk():
    env = gym.make(env_name)
    env.seed(_seed + rank)
    if log_dir is not None:
      env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
    return env
  return _thunk