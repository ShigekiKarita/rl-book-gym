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
