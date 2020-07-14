from gym import spaces
from gym.utils import seeding
import gym
import numpy as np

# aim to hit the target at 45°
HI_ANGLE = 60.0
LO_ANGLE = 20.0


class Fail_v1 (gym.Env):
    metadata = {"render.modes": ["human"]}
    reward_range = (-10.0, 90.0)


    def __init__ (self):
        # NB: the Box bounds for `action_space` must range [-1.0, 1.0]
        # or sampling breaks during rollout
        lo = np.float32(40)
        hi = np.float32(50)

        self.action_space = spaces.Box(
            lo,
            hi,
            shape=(1,)
            )

        self.observation_space = spaces.Box(
            0.0,
            1.0,
            shape=(1,)
            )

        self.np_random = None
        self.reset()


    def reset (self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.seed()

        pos = self.np_random.random()
        self.state = [ pos ]

        self.reward = -100.0
        self.done = False
        self.info = {}

        return self.observation_space.sample()


    def step (self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : array[float]

        Returns
        -------
        observation, reward, done, info : tuple
            observation (object) :
                an environment-specific object representing your observation of
                the environment.

            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.

            done (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)

            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if self.done:
            print("episode done")
            return [self.state, self.reward, self.done, self.info]

        else:
            assert self.action_space.contains(action)

            degree = float(action[0])
            last_pos = self.state[0]

            ## TODO: cal `pos`

            pos = np.sin(np.deg2rad(degree))
            miss = abs(pos - np.sin(np.deg2rad(45.0)))

            self.state[0] = round(pos, 4)
            self.info["action"] = round(degree, 4)
            self.info["miss"] = round(miss, 4)

            self.render()

        if miss <= 0.01:
            # good enough
            self.reward = 90.0
            self.done = True;
        else:
            # reward is the "nearness" of the blast destroying the target
            self.reward = round(-90.0 * miss)

        return [self.state, self.reward, self.done, self.info]


    def render (self, mode="human"):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
        """
        print("location:", self.state, self.reward, self.done, self.info)


    def seed (self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def close (self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass
