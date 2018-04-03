#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pickle

import sonic_on_ray
import ray
from ray.rllib.agent import get_agent_class
from ray.tune.registry import register_env

EXAMPLE_USAGE = ('example usage:\n'
                 './rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run=DQN '
                 '--steps=1000000 --out=rollouts.pkl')

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Roll out a reinforcement learning agent '
                'given a checkpoint.', epilog=EXAMPLE_USAGE)

parser.add_argument(
    'checkpoint', type=str, help='Checkpoint from which to roll out.')
required_named = parser.add_argument_group('required named arguments')
required_named.add_argument(
    '--run', type=str, required=True,
    help='The algorithm or model to train. This may refer to the name '
         'of a built-on algorithm (e.g. RLLib\'s DQN or PPO), or a '
         'user-defined trainable function or class registered in the '
         'tune registry.')
parser.add_argument(
    '--no-render', default=False, action='store_const', const=True,
    help='Surpress rendering of the environment.')
parser.add_argument(
    '--steps', default=None, help='Number of steps to roll out.')
parser.add_argument(
    '--out', default=None, help='Output filename.')
parser.add_argument(
    '--config', default='{}', type=json.loads,
    help='Algorithm-specific configuration (e.g. env, hyperparams). '
         'Surpresses loading of configuration from checkpoint.')

if __name__ == '__main__':
    args = parser.parse_args()

    if not args.config:
        # Load configuration from file
        config_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_dir, 'params.json')
        with open(config_path) as f:
            args.config = json.load(f)

    ray.init()

    env_name = 'sonic_env'
    register_env(env_name, lambda config: sonic_on_ray.make(
                               game='SonicTheHedgehog-Genesis',
                               state='GreenHillZone.Act1'))

    cls = get_agent_class(args.run)
    agent = cls(env=env_name, config=args.config)

    num_steps = int(args.steps)

    # This currently only works with PPO.
    env = agent.local_evaluator.env

    if args.out is not None:
        rollouts = []
    steps = 0
    while steps < (num_steps or steps + 1):
        if args.out is not None:
            rollout = []
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            action = agent.compute_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_total += reward
            if not args.no_render:
                env.render()
            if args.out is not None:
                rollout.append([state, action, next_state, reward, done])
            steps += 1
            state = next_state
        if args.out is not None:
            rollouts.append(rollout)
        print('Episode reward', reward_total)
    if args.out is not None:
        pickle.dump(rollouts, open(args.out, 'wb'))
