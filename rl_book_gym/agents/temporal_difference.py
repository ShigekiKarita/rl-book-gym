from .monte_calro import MonteCarloControler


class AbstractTemporalDifferenceController(MonteCarloControler):
    def __init__(self, step_size, actions, behaviour_policy=None):
        super().__init__(actions, behaviour_policy)
        self.step_size = step_size

    def update_step(self, next_state, state, action, reward, weight):
        raise NotImplementedError()

    def update_episode(self, env):
        raise NotImplementedError()


class SarsaControler(AbstractTemporalDifferenceController):
    """6.5 Sarsa: On-policy TD Control"""

    def init_if_unseen(self, state):
        # TODO: reconsider the initial value
        if state not in self.value_dict:
            self.value_dict[state] = dict()
        for action in self.actions:
            if action not in self.value_dict[state]:
                self.value_dict[state][action] = 0.0 # 1.0 / len(self.actions)


    def update_step(self, next_value, next_state, state, next_action, action, reward):
        self.init_if_unseen(state)
        value = self.value_dict[state][action]
        diff = reward + self.discount * next_value - value
        self.value_dict[state][action] += self.step_size * diff

    def update_episode(self, env):
        obs = env.reset()
        done = False
        cumsum_reward = 0.0
        prev_act = None
        while not done:
            prev_obs = tuple(obs)
            b_act, b_value = self.behaviour_policy(prev_obs)
            obs, reward, done, info = env.step(b_act)
            obs = tuple(obs)
            cumsum_reward = reward + self.discount * cumsum_reward
            if prev_act is not None:
                self.update_step(b_value, obs, prev_obs, b_act, prev_act, reward)
            prev_act = b_act
        return cumsum_reward


class QLearningControler(AbstractTemporalDifferenceController):
    """6.5 Q-Learning: Off-policy TD Control

    NOTE:
    As off-policy method, you can specify independent behaviour policy
    but its default behaviour is eps-greedy policy.
    """

    def update_step(self, next_state, state, action, reward):
        self.init_if_unseen(state)
        g_action, g_value = self.greedy_policy(next_state)
        value = self.value_dict[state][action]
        diff = reward + self.discount * g_value - value
        self.value_dict[state][action] += self.step_size * diff

    def update_episode(self, env):
        obs = env.reset()
        done = False
        cumsum_reward = 0.0
        while not done:
            prev_obs = tuple(obs)
            b_act, b_value = self.behaviour_policy(prev_obs)
            obs, reward, done, info = env.step(b_act)
            obs = tuple(obs)
            cumsum_reward = reward + self.discount * cumsum_reward
            self.update_step(obs, prev_obs, b_act, reward)
        return cumsum_reward
