"""
test code of enviroments
"""


def test_black_jack_score():
    from rl_book_gym.envs.blackjack import score
    assert(score([2, 3]) == 5)
    assert(score([10, 1]) == 21)
    assert(score([10, 1, 1]) == 12)


def test_blackjack_env():
    from rl_book_gym.envs.blackjack import Env, Action
    env = Env()
    env.seed(0)
    d = False
    while not d:
        o, r, d, _ = env.step(Action.hit)
        print(o, r, d)
    env.reset()


def test_windy_gridworld():
    from rl_book_gym.envs.windy_gridworld import Env, Action
    from rl_book_gym.agents.temporal_difference import SarsaControler, QLearningControler
    env = Env()
    env.seed(0)
    agent = QLearningControler(step_size=0.5, actions=Action)
    agent.eps = 0.1
    agent.discount = 0.99
    ret = 0
    for i in range(80000):
        ret += agent.update_episode(env)
        goal = all(env.goal == env.position)
        if goal:
            print("goaled")
        if i % 1000 == 0:
            print(i, ret / (i + 1), goal)

test_windy_gridworld()
