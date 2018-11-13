import numpy


class MonteCarloControler:
    """5.7 Off-policy Monte Carlo Control

    NOTE:
    As off-policy method, you can specify independent behaviour policy
    but its default behaviour is eps-greedy policy.
    """
    def __init__(self, actions, behaviour_policy=None):
        self.value_dict = dict()
        self.sum_weight_dict = dict()
        self.eps = 0.1
        self.discount = 0.9
        self.actions = actions
        if behaviour_policy is None:
            self.behaviour_policy = self.eps_greedy_policy

    def init_if_unseen(self, state):
        # TODO: reconsider the initial value
        if state not in self.value_dict:
            self.value_dict[state] = dict()
            self.sum_weight_dict[state] = dict()
        for action in self.actions:
            if action not in self.value_dict[state]:
                self.value_dict[state][action] = 0
                self.sum_weight_dict[state][action] = 0

    def update_step(self, state, action, cumsum_reward, weight):
        self.init_if_unseen(state)
        self.sum_weight_dict[state][action] += weight
        scale = weight / self.sum_weight_dict[state][action]
        diff = cumsum_reward - self.value_dict[state][action]
        self.value_dict[state][action] += scale * diff

    def update_episode(self, env):
        obs = env.reset()
        done = False
        goal = False
        cumsum_reward = 0.0
        weight = 1
        while not done:
            prev_obs = tuple(obs)
            b_act, b_prob = self.behaviour_policy(prev_obs)
            obs, reward, done, info = env.step(b_act)
            obs = tuple(obs)
            cumsum_reward = reward + self.discount * cumsum_reward
            self.update_step(prev_obs, b_act, cumsum_reward, weight)
            if b_act != self.greedy_policy(prev_obs)[0]:
                break
            if reward == Reward.finish:
                goal = True
            weight /= b_prob
        return cumsum_reward, goal

    def greedy_policy(self, state):
        self.init_if_unseen(state)
        a = max(self.value_dict[state].items(),
                key=lambda kv: kv[1])[0]
        return a, self.value_dict[state][a]

    def eps_greedy_policy(self, state):
        g_action, g_prob = self.greedy_policy(state)
        if numpy.random.rand() > self.eps:
            return g_action, 1 - self.eps
        f = [a for a in self.actions if a != g_action]
        return f[numpy.random.randint(len(f))], self.eps

    def value(self, state, action):
        self.init_if_unseen(state)
        return self.value_dict[state][action]
