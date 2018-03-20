from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import common
import ray
from ray.tune import grid_search, run_experiments
from ray.tune.registry import register_env


env_name = "sonic_env"
register_env(env_name, lambda config: common.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1'))

ray.init()

run_experiments({
  "sonic-ppo": {
    "run": "PPO",
    "env": "sonic_env",
    "resources": {
      "gpu": 2,  # note, keep this in sync with 'devices' config value
      "cpu": lambda spec: spec.config.num_workers,  # one cpu per worker
    },
    "config": {
       # grid search over learning rate
       "sgd_stepsize": grid_search([1e-4, 5e-5, 1e-5, 5e-6]),

       # fixed params for everything else
       "timesteps_per_batch": 40000,
       "min_steps_per_task": 100,
       "num_workers": 4,
       "gamma": 0.99,
       "lambda": grid_search([0,93, 0.95, 0.97, 0.99]),
       "clip_param": grid_search([0.1, 0.2]),
       "num_sgd_iter": grid_search([20, 30]),
       "sgd_batchsize": 4096,
       "use_gae": True,
       "horizon": 4000,
       "devices": ["/gpu:0", "/gpu:1"],
       "tf_session_args": {
         "gpu_options": {"allow_growth": True}
       },
     },
   },
})
