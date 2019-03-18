import numpy as np
import random


ROCK, PAPER, SCISSORS = 0, 1, 2
num_actions = 3

opp_strategy = np.array([0.4, 0.3, 0.3])


def get_strategy(strategy_sum, regret_sum):
    strategy = np.maximum(regret_sum, 0)
    normalizing_sum = np.sum(strategy)

    if normalizing_sum > 0:
        strategy /= normalizing_sum
    else:
        strategy = np.ones(num_actions) / num_actions
    strategy_sum += strategy

    return strategy


def get_avg_strategy(strategy_sum):
    normalizing_sum = np.sum(strategy_sum)

    if normalizing_sum > 0:
        avg_strategy = strategy_sum / normalizing_sum
    else:
        avg_strategy = np.ones(num_actions) / num_actions
    return avg_strategy


def get_action(strategy):
    return np.searchsorted(np.cumsum(strategy), random.random())


def train(iterations):
    regret_sum = np.zeros(num_actions)
    strategy_sum = np.zeros(num_actions)
    action_utility = np.zeros(num_actions)

    for i in range(iterations):
        strategy = get_strategy(strategy_sum, regret_sum)
        my_action = get_action(strategy)
        other_action = get_action(opp_strategy)

        action_utility[other_action] = 0
        action_utility[(other_action + 1) % num_actions] = 1
        action_utility[(other_action - 1) % num_actions] = -1

        regret_sum += action_utility - action_utility[my_action]
    return strategy_sum


print(get_avg_strategy(train(100000)))
