import matplotlib.pyplot as plt
import json

POS_PATH = 'pos/'
FIG_PATH = 'fig/'


def plt_dl(tag: str):
    with open(POS_PATH + tag + '.json', encoding='utf-8') as fp:
        d = json.load(fp)
        logs = d['logs']

        epochs = []
        train_dls = []
        validate_dls = []
        test_dls = []
        for log in logs:
            epochs.append(log['epoch'])
            train_dls.append(log['train_rmsd_metric'])
            validate_dls.append(log['evaluate_rmsd_metric'])
            test_dls.append(log['test_rmsd_metric'])

        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_dls, label='train', c='red')
        plt.plot(epochs, validate_dls, label='validate', c='blue')
        plt.plot(epochs, test_dls, label='test', c='green')
        plt.legend()
        plt.savefig(FIG_PATH + tag + '.png')
        plt.close()


def plt_lt(tag: str, xs: list, ys: list, y2s: list):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    n = len(xs)
    ticks = xs
    # ticks = list(range(n))
    # plt.xticks(ticks, xs)
    ax.plot(ticks, ys, label='loss', c='red')
    ax2 = plt.twinx()
    ax2.plot(ticks, y2s, label='time', c='blue')
    ax.set_ylim(2.3, 3.1)
    ax2.set_ylim(3, 10)
    ax.set_xlabel('Layers')
    ax.set_ylabel('Distance Loss (e-3)')
    ax2.set_ylabel('Process Time (e+4 s)')
    ax.legend(loc=2)
    ax.grid()
    ax2.legend(loc=0)
    plt.savefig(FIG_PATH + tag + '.png')


def plt_l(tag: str, xs: list, ys: list):
    plt.figure(figsize=(4, 4))
    n = len(xs)
    ticks = xs
    # ticks = list(range(n))
    # plt.xticks(ticks, xs)
    plt.plot(ticks, ys, c='green')
    plt.ylim(2, 7)
    plt.xlabel('PQ Dimension')
    plt.ylabel('Distance Loss (e-3)')
    plt.grid()
    plt.savefig(FIG_PATH + tag + '.png')


plt_lt('Hamiltonian Engine Layers', [0, 4, 10, 15, 20, 30, 40],
       [3.046, 2.761, 2.670, 2.582, 2.689, 2.389, 2.551], [3.8548, 3.9611, 5.0986, 4.4620, 5.9848, 6.3564, 8.8961])
plt_l('PQ Dimension', [3, 8, 16, 32, 64],
      [6.887, 5.025, 2.986, 2.689, 2.652])
