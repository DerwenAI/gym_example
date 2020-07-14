#!/usr/bin/env python
# encoding: utf-8

import gym
import gym_example


N_ITER = 10

           
def main ():
    env = gym.make("example-v0")
    #env = gym.make("CartPole-v1")
    env.reset()

    for i in range(N_ITER):
        action = env.action_space.sample()
        print("action", action)
        state, reward, done, info = env.step(action)

        print(state, reward, done, info)

        if done:
            print("done @ step {}".format(i))
            break


if __name__ == "__main__":
    main()
