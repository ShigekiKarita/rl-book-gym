from enum import IntEnum
import numpy


class Action(IntEnum):
    up = 0
    down = 1
    right = 2
    left = 3


class Reward(IntEnum):
    finish = 1
    not_finish = -1


class Env:
    """Example 6.5: Windy Gridworld Env"""
    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(self, seed=None, itype=numpy.int64):
        self.rng = numpy.random.RandomState(seed)
        self.start = numpy.array([0, 3], dtype=itype)
        self.wind = numpy.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0], dtype=itype)
        self.goal = numpy.array([0, 7], dtype=itype)
        self.bound = numpy.array([6, 9], dtype=itype)
        self.position = self.start[:]

    def is_out(self):
        x, y = self.position
        return x < 0 or self.bound[0] < x or y < 0 or self.bound[1] < y

    def step(self, action):
        """
        Parameters
        ----------
            action: Action

        Returns
        -------
            A tuple of (observation, reward, done, info)
            observation: numpy.ndarray([x position, y position])
            reward: int
                The rewards are −1 for each step until reaching the goal
            done: bool
                true only if the car crosses the finish line
            info: dict
                empty
        """
        if self.is_out():
            return self.position[:], Reward.not_finish, True, dict()
        self.position += self.wind[self.position[0]]
        if action == Action.left:
            self.position[0] -= 1
        elif action == Action.right:
            self.position[0] += 1
        elif action == Action.up:
            self.position[1] += 1
        elif action == Action.down:
            self.position[1] -= 1

        if all(self.position == self.goal):
            reward = Reward.finish
            done = True
        else:
            reward = Reward.not_finish
            done = self.is_out()
        observation = self.position[:]
        info = dict()
        return observation, reward, done, info

    def reset(self):
        self.position = self.start
        return self.position[:]

    def seed(self, seed=None):
        """
        Args:
            seed (int): numpy RandomState seed
        """
        self.rng.seed(seed)
