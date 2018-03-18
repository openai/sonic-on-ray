import ray
from ray.tune.registry import get_registry, register_env
from baselines.common.atari_wrappers import FrameStack
from ray.rllib import ppo
import gym
import gym.spaces as spaces
import retro
import numpy as np
import cv2


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 80
        self.height = 80
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


def make(game, state, stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    env = retro.make(game, state)
    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    return env

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01


env_name = "sonic_env"
register_env(env_name, lambda config: make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1'))

ray.init()

config = ppo.DEFAULT_CONFIG.copy()

config.update({
  "timesteps_per_batch": 40000,
  "min_steps_per_task": 100,
  "num_workers": 32,
  "gamma": 0.99,
  "lambda": 0.95,
  "clip_param": 0.1,
  "num_sgd_iter": 30,
  "sgd_batchsize": 4096,
  "sgd_stepsize": 2e-4,
  "use_gae": False,
  "horizon": 4000,
  "devices": ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5", "/gpu:6", "gpu:7"],
  "tf_session_args": {
    "gpu_options": {"allow_growth": True}
  }
  # "tf_debug_inf_or_nan": True
})

alg = ppo.PPOAgent(config=config, env=env_name, registry=get_registry())

for i in range(1000):
    result = alg.train()
    print("result = {}".format(result))

    if i % 10 == 0:
        checkpoint = alg.save()
        print("checkpoint saved at", checkpoint)
