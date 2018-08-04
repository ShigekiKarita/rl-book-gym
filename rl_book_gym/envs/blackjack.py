from enum import IntEnum
import numpy


class Action(IntEnum):
    hit = 0
    stick = 1


class Reward(IntEnum):
    win = 1
    draw = 0
    lose = -1


def score(cards):
    s = sum(cards)
    if s <= 11 and 1 in cards:
        s += 10
    return s


class Env:
    """Black Jack Env"""
    def __init__(self, seed=None):
        super().__init__()
        self.rng = numpy.random.RandomState(seed)
        self.player_card = []
        self.dealer_card = []
        self.stick_score = 17

    def new_card(self):
        card = self.rng.randint(1, 13)
        return min(card, 10)

    def step(self, action: Action):
        # player action
        if action == Action.hit:
            self.player_card.append(self.new_card())
        # dealer action
        dealer_hit = score(self.dealer_card) < self.stick_score
        if dealer_hit:
            self.dealer_card.append(self.new_card())

        dealer_score = score(self.dealer_card)
        player_score = score(self.player_card)
        dealer_bust = dealer_score > 21
        player_bust = player_score > 21
        done = (action == Action.stick and not dealer_hit) or player_bust or dealer_bust
        if done:
            if not player_bust and (dealer_bust or player_score > dealer_score):
                reward = Reward.win
            elif (player_bust and dealer_bust) or (player_score == 21 and dealer_score == 21):
                reward = Reward.draw
            else:
                reward = Reward.lose
        else:
            reward = Reward.draw

        # NOTE: in the original rule, only one dealer card is available
        # NOTE: tuple() is deep copy
        obs = dict(player_card=tuple(self.player_card),
                   dealer_card=tuple(self.dealer_card),
                   player_score=player_score,
                   dealer_score=dealer_score)
        info = dict()
        return obs, reward, done, info

    def reset(self):
        self.player_card = []
        self.dealer_card = []

    def close(self):
        return

    def seed(self, seed=None):
        self.rng.seed(seed)
