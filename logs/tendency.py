import matplotlib.pyplot as plt
import json
import seaborn as sns

POS_PATH = 'pos/'
FIG_PATH = 'fig/'

font1 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 13,
}


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
    ax.plot(ticks, ys, label='loss', c=sns.xkcd_rgb['pale red'])
    ax2 = plt.twinx()
    ax2.plot(ticks, y2s, label='time', c=sns.xkcd_rgb['denim blue'])
    ax.set_ylim(4.8, 5.7)
    ax2.set_ylim(3, 11)
    ax.set_ylabel('Distance Loss (e-2)', font1)
    ax2.set_ylabel('Process Time (e+4 s)', font1)
    ax.yaxis.label.set_color(color=sns.xkcd_rgb['pale red'])
    ax2.yaxis.label.set_color(color=sns.xkcd_rgb['denim blue'])
    ax.tick_params(axis='y', colors=sns.xkcd_rgb['pale red'])
    ax2.tick_params(axis='y', colors=sns.xkcd_rgb['denim blue'])
    ax.legend(loc=2)
    ax.grid()
    ax2.legend(loc=1)
    plt.savefig(FIG_PATH + tag + '.png')
    plt.close(fig)


def plt_lt2(tag: str, xs: list, ys: list, y2s: list):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    n = len(xs)
    ticks = xs
    # ticks = list(range(n))
    # plt.xticks(ticks, xs)
    ax.plot(ticks, ys, label='loss', c=sns.xkcd_rgb['pale red'])
    ax2 = plt.twinx()
    ax2.plot(ticks, y2s, label='time', c=sns.xkcd_rgb['denim blue'])
    ax.set_ylim(5.0, 9.0)
    ax2.set_ylim(4.5, 7)
    ax2.set_yticks(range(5, 8))
    ax.set_ylabel('Distance Loss (e-2)', font1)
    ax2.set_ylabel('Process Time (e+4 s)', font1)
    ax.yaxis.label.set_color(color=sns.xkcd_rgb['pale red'])
    ax2.yaxis.label.set_color(color=sns.xkcd_rgb['denim blue'])
    ax.tick_params(axis='y', colors=sns.xkcd_rgb['pale red'])
    ax2.tick_params(axis='y', colors=sns.xkcd_rgb['denim blue'])
    ax.legend(loc=2)
    ax.grid()
    ax2.legend(loc=0)
    plt.savefig(FIG_PATH + tag + '.png')
    plt.close(fig)


def plt_bar(tag: str, xs: list, ys: list, y2s: list):
    fig = plt.figure(figsize=(6.2, 4))
    ax1 = fig.add_subplot(111)

    # 柱形的宽度
    width = 0.3

    # 柱形的间隔
    x1_list = []
    x2_list = []
    for i in range(len(xs)):
        x1_list.append(i)
        x2_list.append(i + width)
    x1_list.reverse()
    x2_list.reverse()

    # 绘制柱形图1
    b1 = ax1.barh(x2_list, ys, height=width, label='Distance Loss', color=sns.xkcd_rgb['pale red'])

    # 绘制柱形图2---双Y轴
    ax2 = ax1.twiny()
    b2 = ax2.barh(x1_list, y2s, height=width, label='Kabsch-RMSD', color=sns.xkcd_rgb['denim blue'])

    # 坐标轴标签设置
    plt.yticks([(x1 + x2) / 2 for x1, x2 in zip(x1_list, x2_list)], xs)
    ax1.set_xlabel('Distance Loss (e-2)', font1)
    ax2.set_xlabel('Kabsch-RMSD', font1)
    ax1.set_xlim(4.0, 11.0)
    ax2.set_xlim(1.0, 2.2)

    # x轴标签旋转
    # ax1.set_yticklabels(ax1.get_yticklabels(), rotation=25)

    # 双Y轴标签颜色设置
    ax1.xaxis.label.set_color(b1[0].get_facecolor())
    ax2.xaxis.label.set_color(b2[0].get_facecolor())

    # 双Y轴刻度颜色设置
    ax1.tick_params(axis='x', colors=b1[0].get_facecolor())
    ax2.tick_params(axis='x', colors=b2[0].get_facecolor())

    # 图例设置
    plt.legend(handles=[b1, b2])

    # 网格设置
    plt.grid('off')

    plt.savefig(FIG_PATH + tag + '.png')
    plt.close(fig)


plt_lt('HamiltonianEngineLayers', [0, 4, 10, 15, 20, 30, 40, 50],
       [5.519, 5.255, 5.167, 5.081, 5.186, 4.888, 5.051, 5.031],
       [3.8548, 3.9611, 5.0986, 4.4620, 5.9848, 6.3564, 8.8961, 9.4736])
plt_lt2('PQDimension', [3, 8, 16, 32, 64],
        [8.299, 7.089, 5.464, 5.186, 5.150],
        [5.5966, 6.1525, 6.1770, 6.1749, 6.1707])
plt_bar('ConformationPrediction', ['Complete', 'w/o ADJ', 'w/o F', 'w/o H', 'w/o LSTM',
                                   # 'MPNN', 'RDKit'
                                   ], [5.186, 7.746, 5.227, 5.519, 10.871,
                                       # 8.620, 7.519
                                       ], [1.384, 1.087, 1.389, 1.442, 2.039,
                                           # 1.708, 1.649
                                           ])
