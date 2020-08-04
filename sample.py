#!/usr/bin/env python
# encoding: utf-8

import gym
import gym_example


def run_one_episode (env, verbose=False):
    env.reset()
    sum_reward = 0

    for i in range(env.MAX_STEPS):
        action = env.action_space.sample()

        if verbose:
            print("action:", action)

        state, reward, done, info = env.step(action)
        sum_reward += reward

        if verbose:
            env.render()

        if done:
            if verbose:
                print("done @ step {}".format(i))

            break

    if verbose:
        print("cumulative reward", sum_reward)

    return sum_reward


def main ():
    # first, create the custom environment and run it for one episode
    env = gym.make("example-v0")
    sum_reward = run_one_episode(env, verbose=True)

    # next, calculate a baseline of rewards based on random actions
    # (no policy)
    history = []

    for _ in range(10000):
        sum_reward = run_one_episode(env, verbose=False)
        history.append(sum_reward)

    avg_sum_reward = sum(history) / len(history)
    print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))


if __name__ == "__main__":
    main()
