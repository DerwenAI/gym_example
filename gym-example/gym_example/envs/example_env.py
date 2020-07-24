from gym import spaces
from gym.utils import seeding
import gym

DEBUG = False # True


# possible actions
MOVE_LF = 0
STAY = 1
MOVE_RT = 2

# possible positions
LF_MOST = 1
GOAL = 5
RT_MOST = 9

# aim to land on the GOAL position within MAX_COUNT steps
MAX_COUNT = 10

# possible rewards
REWARD_AWAY = -2
REWARD_NOOP = -1
REWARD_GOAL = MAX_COUNT


class Example_v0 (gym.Env):
    metadata = {"render.modes": ["human"]}
    reward_range = (REWARD_AWAY, REWARD_GOAL)


    def __init__ (self):
        # the action space ranges [0, 1, 2] where:
        #  `0` move left
        #  `1` stay in position
        #  `2` move right
        self.action_space = spaces.Discrete(3)

        # NB: Ray throws exceptions for any `0` value Discrete
        # observations so we'll make position a 1's based value
        self.observation_space = spaces.Discrete(RT_MOST + 1)

        # enumerate the possible positions, then chose on reset()
        self.init_positions = list(range(LF_MOST, RT_MOST))
        self.init_positions.remove(GOAL)

        # NB: change this to guarantee the same sequence of
        # pseudorandom numbers each time (e.g., for debugging)
        self.seed()

        self.reset()


    def reset (self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.position = self.np_random.choice(self.init_positions)
        self.count = 0

        # for this environment, state is simply the position
        self.state = self.position
        self.reward = 0
        self.done = False
        self.info = {}

        return self.state


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
        global DEBUG

        if self.done:
            # code should never reach this point
            print("EPISODE DONE!!!")

        elif self.count == MAX_COUNT:
            self.done = True;

            if DEBUG:
                print("episode done")

        else:
            assert self.action_space.contains(action)

            if action == MOVE_LF:
                if self.position == LF_MOST:
                    # invalid
                    self.reward = REWARD_AWAY
                else:
                    self.position -= 1

                    if self.position == GOAL:
                        # on goal now
                        self.reward = REWARD_GOAL
                        self.done = 1
                    elif self.position < GOAL:
                        # moving away from goal
                        self.reward = REWARD_AWAY
                    else:
                        # moving toward goal
                        self.reward = REWARD_NOOP

            elif action == MOVE_RT:
                if self.position == RT_MOST:
                    # invalid
                    self.reward = REWARD_AWAY
                else:
                    self.position += 1

                    if self.position == GOAL:
                        # on goal now
                        self.reward = REWARD_GOAL
                        self.done = 1
                    elif self.position > GOAL:
                        # moving away from goal
                        self.reward = REWARD_AWAY
                    else:
                        # moving toward goal
                        self.reward = REWARD_NOOP

            elif action == STAY:
                if self.position == GOAL:
                    # code should never reach here
                    print("STUCK ON GOAL!!!")
                    self.reward = REWARD_GOAL
                    self.done = 1
                else:
                    # not moving toward goal
                    self.reward = REWARD_AWAY


            self.count += 1
            self.state = self.position

            self.info["count"] = self.count
            self.info["action"] = action
            self.info["dist"] = GOAL - self.position

            if DEBUG:
                self.render()

        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("STATE IS WRONG", self.state)

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
        print("position: {:2d}  reward: {:2d}  info: {}".format(self.state, self.reward, self.info))


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
