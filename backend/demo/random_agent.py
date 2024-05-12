import random


def random_action(state):
    temp = state.flatten()
    valid_actions = [i for i in range(len(temp)) if temp[i] == 0]
    return random.choice(valid_actions)