import os

import gym
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

from sonic_tools.sonic_util import make_sonic_train, make_sonic_test

import numpy as np

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import pybullet_envs
    import roboschool
except ImportError:
    pass


def make_env_test():
    def _thunk():
        # if env_id.startswith("dm"):
        #     _, domain, task = env_id.split('.')
        #     env = dm_control2gym.make(domain_name=domain, task_name=task)
        # else:
        #     env = gym.make(env_id)

        env = make_sonic_test()

        # is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        # if is_atari:
        #     env = make_atari(env_id)
        #env.seed(seed + rank)

        #if log_dir is not None:
        #    env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        # if is_atari:
        #     env = wrap_deepmind(env)
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape

        #TODO Verify this guy
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3, 4]:
            env = WrapPyTorch(env)
        return env

    return _thunk

def make_env_train(env_id, seed, rank, log_dir):
    def _thunk():
        # if env_id.startswith("dm"):
        #     _, domain, task = env_id.split('.')
        #     env = dm_control2gym.make(domain_name=domain, task_name=task)
        # else:
        #     env = gym.make(env_id)

        env = make_sonic_train(env_id=env_id)

        # is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        # if is_atari:
        #     env = make_atari(env_id)
        env.seed(seed + rank)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        # if is_atari:
        #     env = wrap_deepmind(env)
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        print(obs_shape)

        #TODO Verify this guy
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3, 4]:
            env = WrapPyTorch(env)
        return env

    return _thunk

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]]
        )

    def _observation(self, observation):
        # TODO: Check this workaround
        observation = np.squeeze(np.array(observation._frames))
        return observation.transpose(2, 0, 1)
