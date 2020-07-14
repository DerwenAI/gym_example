#!/usr/bin/env python
# encoding: utf-8

from gym_example.envs.example_env import Example_v0
from ray.tune.registry import register_env
import gym
import pprint
import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil
#import ray.rllib.agents.sac as sac
import sys
# NB: SACTrainer requires tensorflow_probability so test it here
import tensorflow as tf
#import tensorflow_probability as tfp


CHECKPOINT_ROOT = "tmp/exa"
#SELECT_ENV = "CartPole-v1"
SELECT_ENV = "example-v0"
N_ITER = 1
s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"


def train_policy (agent, path, debug=True, n_iter=N_ITER):
    reward_history = []

    for n in range(n_iter):
        result = agent.train()
        file_name = agent.save(CHECKPOINT_ROOT)

        print(s.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                file_name
                ))


def rollout_actions (agent, env, debug=True, render=True, max_steps=1000, episode_interval=20):
    for step in range(max_steps):
        if step % episode_interval == 0:
            state = env.reset()

        last_state = state
        print("state", state)
        action = agent.compute_action(state)
        state, reward, done, info = env.step(action)

        if debug:
            print("state", last_state, "action", action, "reward", reward)
            print(info)

        if render:
            env.render()

        if done == 1 and reward > 0:
            break


if __name__ == "__main__":
    ## start Ray
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    ## set up directories to log results and checkpoints
    shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    ## configure the environment
    #config = sac.DEFAULT_CONFIG.copy()
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"

    register_env("example-v0", lambda config: Example_v0())

    # train a policy with RLlib using SAC
    #agent = sac.SACTrainer(config, env=SELECT_ENV)
    agent = ppo.PPOTrainer(config, env=SELECT_ENV)
    train_policy(agent, CHECKPOINT_ROOT)

    ## examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())
    sys.exit(0)

    # apply the trained policy in a rollout
    agent.restore(checkpoint_path)
    env = gym.make(SELECT_ENV)

    rollout_actions(agent, env)
