import numpy
from enum import IntEnum


class Action(IntEnum):
    up = 1
    keep = 0
    down = -1


class Reward(IntEnum):
    finish = 1
    not_finish = -1


class Map(IntEnum):
    out_side = 0
    in_side = 1
    start = 2
    finish = 3
    car = 4


class Env:
    """Exercise 5.10:  Racetrack (programming)
Consider driving a race car around a turn like those shown in Figure 5.5.
You want to go as fast as possible, but not so fast as to run off the track.

In our simplified racetrack, the car is at one of a discrete set of grid positions,
the cells in the diagram.

The velocity is also discrete, a number of grid cells moved
horizontally and vertically per time step.

The actions are increments to the velocity components.
Each may be changed by +1, −1, or 0 in each step, for a total of nine  (3×3)  actions.

Both  velocity  components  are  restricted  to  be  nonnegative and less  than  5,
and they cannot both be zero except at the starting line.

Each episode begins in one of the randomly selected start states with both velocity components zero and ends when the car crosses the finish line.
The rewards are −1 for each step until the car crosses the finish line.

If the car hits the track boundary, it is moved back to a random position on the starting line,
both velocity components are reduced to zero, and the episode continues.

Before updating the car’s location at each time step,
check to see if the projected path of the car intersects the track boundary.

If it intersects the finish line, the episode ends;
if it intersects anywhere else, the car is considered to have hit the track boundary and is sent
back to the starting line.

To make the task more challenging, with probability 0.1 at each time step
the velocity increments are both zero, independently of the intended increments.

Apply a Monte Carlo control method to this task to compute the optimal policy from each starting state.
Exhibit several trajectories following the optimal policy (but turn the noise off for these trajectories)
"""
    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(self, csv_path, seed=None, drift=0.1, itype=numpy.int64):
        self.rng = numpy.random.RandomState(seed)
        self.drift = drift
        self.map_info = numpy.loadtxt(csv_path, delimiter=",", dtype=itype)
        self.starts = numpy.stack(numpy.where(self.map_info == Map.start)).T
        self.position = None
        # The velocity is also discrete, a number of grid cells moved
        # horizontally and vertically per time step.
        self.velocity = numpy.zeros(2, dtype=itype)
        self.viewer = None

    def __del__(self):
        self.close()

    def choice_start(self):
        i = self.rng.randint(len(self.starts))
        xy = self.starts[i]
        assert self.map_info[xy[0], xy[1]] == Map.start
        return xy

    def update_velocity(self, action):
        """Update self.velocity with respect to an action and random drift.

        If the car hits the track boundary,
        it is moved back to a random position on the starting line,
        both velocity components are reduced to zero, and the episode continues.

        Both velocity components are restricted to be nonnegative and less than 5,
        and they cannot both be zero except at the starting line.
        To make the task more challenging, with probability 0.1 at each time step
        the velocity increments are both zero, independently of the intended increments.

        Parameters
        ----------
            action: numpy.ndarray
                shape=[2], dtype=numpy.int64
        """
        v = self.velocity + action
        a = abs(v)
        if all(a == 0) or any(a >= 5) or self.rng.rand() < self.drift:
            return
        self.velocity = v

    def update_position(self):
        """Update position with respect to the intersects between current and next position

        Before updating the car’s location at each time step,
        check to see if the projected path of the car intersects the track boundary.
        If it intersects the finish line, the episode ends;
        if it intersects anywhere else, the car is considered to have hit the track boundary
        and is sent back to the starting line.

        Parameters
        ----------
            reward: int
            done: bool
        """
        next_position = self.position + self.velocity
        x, y = next_position
        px, py = self.position
        ax, bx = max(0, min(px, x)), min(max(px, x), self.map_info.shape[0])
        ay, by = max(0, min(py, y)), min(max(py, y), self.map_info.shape[1])
        intersects = self.map_info[ax:bx+1, ay:by+1]
        # FIXME: this may be too loose finish condition
        # to avoid intersection anywhere else finish line
        if Map.finish in intersects:
            return Reward.finish, True
        if Map.out_side in intersects:
            self.reset()
        else:
            self.position = next_position
        return Reward.not_finish, False

    def render(self, mode="human", close=False):
        import matplotlib.cm
        from .viewer import SimpleImageViewer
        cmap = matplotlib.cm.get_cmap()
        data = self.map_info.copy()
        if self.position is not None:
            x, y = self.position
            data[x, y] = Map.car
        ret = (data - numpy.min(data))/ (numpy.max(data) - numpy.min(data))
        ret = cmap(ret)[:, :, :3]  # omit (rgb)a
        if mode == "human":
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow((ret * 255).repeat(8, 0).repeat(8, 1).astype(numpy.uint8))
            if close:
                self.viewer.close()
            return self.viewer.isopen
        else:
            return ret

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def step(self, action):
        """
        Parameters
        ----------
            action: numpy.ndarray
                [x action, y action]
                The actions are increments to the velocity components.
                Each may be changed by +1, −1, or 0 in each step,
                for a total of nine  (3×3)  actions.

        Returns
        -------
            A tuple of (observation, reward, done, info)
            observation: numpy.ndarray
                [x position, y position, x velocity, y velocity]
            reward: int
                The rewards are −1 for each step until the car crosses the finish line.
            done: bool
                true only if the car crosses the finish line
            info: dict
                empty
        """
        self.update_velocity(action)
        if self.position is None:
            self.position = self.choice_start()
            reward = Reward.not_finish
            done = False
        else:
            reward, done = self.update_position()
        observation = numpy.concatenate((self.position, self.velocity))
        info = dict()
        return observation, reward, done, info

    def reset(self):
        self.position = self.choice_start()
        self.velocity[:] = 0

    def seed(self, seed=None):
        """
        Args:
            seed (int): numpy RandomState seed
        """
        self.rng.seed(seed)
