from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonic_on_ray
import ray
from ray.rllib import ppo
from ray.tune.registry import register_env


env_name = 'sonic_env'
# Note that the hyperparameters have been tuned for sonic, which can be used
# run by replacing the below function with:
#
#     register_env(env_name, lambda config: sonic_on_ray.make(
#                                game='SonicTheHedgehog-Genesis',
#                                state='GreenHillZone.Act1'))
#
# However, to try Sonic, you have to obtain the ROM yourself (see then
# instructions at https://github.com/openai/retro/blob/master/README.md).
register_env(env_name,
             lambda config: sonic_on_ray.make(game='Airstriker-Genesis',
                                              state='Level1'))

ray.init()

config = ppo.DEFAULT_CONFIG.copy()

config.update({
    'timesteps_per_batch': 40000,
    'min_steps_per_task': 100,
    'num_workers': 32,
    'gamma': 0.99,
    'lambda': 0.95,
    'clip_param': 0.1,
    'num_sgd_iter': 30,
    'sgd_batchsize': 4096,
    'sgd_stepsize': 5e-5,
    'use_gae': True,
    'horizon': 4000,
    'devices': ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3', '/gpu:4', '/gpu:5',
                '/gpu:6', 'gpu:7'],
    'tf_session_args': {
        'gpu_options': {'allow_growth': True}
    }
})

alg = ppo.PPOAgent(config=config, env=env_name)

for i in range(1000):
    result = alg.train()
    print('result = {}'.format(result))

    if i % 10 == 0:
        checkpoint = alg.save()
        print('checkpoint saved at', checkpoint)
