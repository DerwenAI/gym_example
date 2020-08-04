#!/usr/bin/env python
# encoding: utf-8

from gym_example.envs.example_env import Example_v0
from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil


def train_policy (agent, path, n_iter=1):
    status = "{:2d}  reward {:6.2f}/{:6.2f}/{:6.2f}  len {:4.2f}  saved {}"
    reward_history = []

    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save(path)

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))

    return chkpt_file


def rollout_actions (agent, env, n_step=1, render=True):
    state = env.reset()
    sum_reward = 0

    for step in range(n_step):
        action = agent.compute_action(state)
        state, reward, done, info = env.step(action)
        sum_reward += reward

        if render:
            env.render()

        if done == 1:
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0


if __name__ == "__main__":
    # start Ray
    ray.init(ignore_reinit_error=True, local_mode=True)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    # init directory in which to save checkpoints
    CHECKPOINT_ROOT = "tmp/exa"
    shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

    # configure the environment
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"

    # register the custom environment
    SELECT_ENV = "example-v0"
    register_env(SELECT_ENV, lambda config: Example_v0())

    # train a policy with RLlib using PPO
    agent = ppo.PPOTrainer(config, env=SELECT_ENV)
    chkpt_file = train_policy(agent, CHECKPOINT_ROOT, n_iter=5)

    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())

    # apply the trained policy in a rollout
    agent.restore(chkpt_file)
    env = gym.make(SELECT_ENV)
    rollout_actions(agent, env, n_step=20)
