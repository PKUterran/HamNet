import numpy as np


def sample(items: list, p1: float, p2: float, p3: float) -> (list, list, list):
    assert 1 - 1e-5 < p1 + p2 + p3 < 1 + 1e-5
    num = len(items)
    items = np.random.permutation(items)
    train_num = int(num * p1)
    validate_num = int(num * p2)
    return items[: train_num], items[train_num: train_num + validate_num], items[train_num + validate_num:]
