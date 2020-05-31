import matplotlib.pyplot as plt
import numpy as np


def plt_multiple_scatter(path: str, masks: list, logits: np.ndarray, target: np.ndarray) -> (np.ndarray, np.ndarray):
    assert logits.shape == target.shape, 'Unmatched: {}, {}.'.format(logits.shape, target.shape)
    for i in range(logits.shape[-1]):
        l = logits[:, i]
        t = target[:, i]
        v = max(np.abs(l).max(), np.abs(t).max()) + 0.5
        plt.figure(figsize=[8, 8])
        plt.xlim((-v, v))
        plt.ylim((-v, v))
        plt.scatter(l, t)
        plt.savefig('{}_{}.png'.format(path, i))
        plt.close()

    dd = np.abs(logits - target)
    d = np.argsort(dd.sum(axis=1))
    worst_ids = d[-5:]
    worst_ds = dd[worst_ids, :]
    best_ids = d[:5]
    best_ds = dd[best_ids, :]
    return best_ids, best_ds, worst_ids, worst_ds
