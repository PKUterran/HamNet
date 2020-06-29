import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.manifold import TSNE
from functools import reduce

TRAJECTORY_DIR = 'graphs/trajectory'


def plt_trajectory(qs: list, us: np.ndarray, vs: np.ndarray, name: str = ''):
    all_q = np.vstack(qs)
    tsne = TSNE(n_components=2)
    tsne.fit(all_q)
    qs = [tsne.fit_transform(q) for q in qs]
    all_q = np.vstack(qs)
    # print(all_q)
    i = len(qs)
    n = qs[0].shape[0]
    s = reduce(lambda x, y: x + y, [[10 * j ** 1.5] * n for j in range(1, i + 1)])
    c = list(range(n)) * i
    # print(s)
    # print(c)

    fig = plt.figure(figsize=[12, 12])
    plt.scatter(all_q[:, 0], all_q[:, 1], s=s, c=c)
    plt.savefig('{}/{}.png'.format(TRAJECTORY_DIR, name))
    plt.close()

    q = qs[-1]
    fig = plt.figure(figsize=[12, 12])
    plt.scatter(q[:, 0], q[:, 1], s=10 * i ** 1.5, c=list(range(n)))
    for u, v in zip(us, vs):
        plt.plot([q[u, 0], q[v, 0]], [q[u, 1], q[v, 1]], color='black', linewidth=5)
    plt.savefig('{}/{}_final.png'.format(TRAJECTORY_DIR, name))
    plt.close()
