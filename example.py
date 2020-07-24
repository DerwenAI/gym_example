#!/usr/bin/env python
# encoding: utf-8

import gym
import gym_example


def main ():
    env = gym.make("example-v0")
    env.reset()
    sum_reward = 0

    for i in range(10):
        action = env.action_space.sample()
        print("action:", action)

        state, reward, done, info = env.step(action)
        sum_reward += reward
        env.render()

        if done:
            print("done @ step {}".format(i))
            break

    print("cumulative reward", sum_reward)


if __name__ == "__main__":
    main()
