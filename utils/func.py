import numpy as np


def re_index(us: np.ndarray, node_mask: np.ndarray):
    d = {}
    cnt = 0
    for i, m in enumerate(node_mask):
        if m:
            d[i] = cnt
            cnt += 1

    return np.array([d[u] for u in us])
